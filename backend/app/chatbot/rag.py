# backend/app/chatbot/rag.py
from datetime import datetime, timedelta, timezone
from dateutil import parser
import logging
logger = logging.getLogger(__name__)
import math
import re
import os
from time import perf_counter
from dotenv import load_dotenv
from .vectorstore import load_vectorstore, similarity_search  
from .llm import load_llm, generate_answer
from rank_bm25 import BM25Okapi
import regex as re2
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from ..utils_text import get_source_from_link

# Φόρτωση .env
load_dotenv()

# Feature flags & ρυθμίσεις
RERANK_BM25_ENABLED = os.getenv("RERANK_BM25_ENABLED", "true").lower() == "true"
EXTRACTIVE_ENABLED  = os.getenv("EXTRACTIVE_ENABLED", "true").lower() == "true"
SNIPPETS_K          = int(os.getenv("SNIPPETS_K", "5"))
SNIPPETS_MIN_SIM    = float(os.getenv("SNIPPETS_MIN_SIM", "0.15"))

# cross-encoder re-rank 
CROSS_ENCODER_ENABLED = os.getenv("CROSS_ENCODER_ENABLED", "false").lower() == "true"
CROSS_ENCODER_MODEL   = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

class ConversationManager:
   
    def __init__(self):
        self.conversation_context = {}
    
    def store_article_context(self, session_id: str, article_doc, user_query: str, answer: str):
        """
        Αποθηκεύει το context του άρθρου για μελλοντικές ερωτήσεις
        """
        self.conversation_context[session_id] = {
            'article': {
                'title': article_doc.metadata.get('title', ''),
                'content': article_doc.page_content[:2000],  
                'source': article_doc.metadata.get('source', ''),
                'published_date': article_doc.metadata.get('published_date', ''),
                'link': article_doc.metadata.get('link', '')
            },
            'original_query': user_query,
            'previous_answer': answer,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_article_context(self, session_id: str):
        """Επιστρέφει το αποθηκευμένο article context"""
        return self.conversation_context.get(session_id)
    
    def clear_context(self, session_id: str):
        """Διαγράφει το context"""
        if session_id in self.conversation_context:
            del self.conversation_context[session_id]

conversation_manager = ConversationManager()

def select_multiple_articles(hits_with_raw, query: str, max_articles: int = 4, max_age_days: int = 60):
    """
    Επιλέγει πολλά άρθρα με VERY RELAXED similarity threshold για περισσότερα results
    """
    current_date = datetime.now(timezone.utc)
    
    print(f" MULTI-ARTICLE SELECTION: seeking {max_articles} articles (max {max_age_days} days old)")
    
    # Φιλτράρισμα και scoring candidates
    candidates = []
    seen_content_previews = set()
    
    for doc, raw_score in hits_with_raw:
        sim = _distance_to_similarity(raw_score)
        
        #  Χαλαρό threshold
        if sim < 0.04:  
            continue

        text = (doc.page_content or "").strip()
        if len(text) < 80: 
            continue

        # Έλεγχος ημερομηνίας
        pub_date = doc.metadata.get("published_date")
        pub_dt = _to_utc(pub_date)
        
        if not pub_dt:
            continue
            
        age_days = (current_date - pub_dt).days
        
        # Απόρριψη παλιών άρθρων
        if age_days > max_age_days:
            continue
        
        # Deduplication με content preview
        content_preview = text[:120].strip().lower()
        if content_preview in seen_content_previews:
            continue
        seen_content_previews.add(content_preview)
        
        # Scoring
        title = doc.metadata.get("title", "").lower()
        source = _get_article_source(doc)
        
        # Βασικό score
        base_score = sim
        
        # Recency bonus
        recency_bonus = _calculate_aggressive_recency_bonus(age_days)
        
        # Query relevance
        relevance_score = 0
        query_terms = query.lower().split()
        title_matches = sum(1 for term in query_terms if len(term) > 2 and term in title)
        relevance_score += title_matches * 0.15  
        
        # Source diversity bonus
        source_bonus = 0.05 if source not in ["Associated Press", "AP News", "Unknown Source"] else 0.0
        
        # Final score
        final_score = base_score + recency_bonus + relevance_score + source_bonus
        
        candidates.append({
            'score': final_score,
            'doc': doc,
            'age_days': age_days,
            'similarity': sim,
            'source': source,
            'title': doc.metadata.get('title', ''),
            'pub_date': pub_date,
            'recency_bonus': recency_bonus
        })
    
    if not candidates:
        print("No suitable articles found for multi-article selection")
        return []
    
    # Ταξινόμηση κατά score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print("TOP CANDIDATES:")
    for i, candidate in enumerate(candidates[:10]):
        title = candidate['title'][:60]
        source = candidate['source']
        score = candidate['score']
        age = candidate['age_days']
        recency = candidate['recency_bonus']
        print(f"  {i+1}. Score: {score:.3f} | Age: {age}d | Recency: {recency:.3f} | {source} | '{title}...'")
    
    # Diversity selection
    selected = []
    sources_used = {}
    seen_titles = set()
    
    for candidate in candidates:
        if len(selected) >= max_articles:
            break
            
        doc = candidate['doc']
        source = candidate['source']
        title = candidate['title']
        age_days = candidate['age_days']
        
        # Απλό deduplication 
        title_lower = title.lower().strip()
        if title_lower in seen_titles:
            continue
        
        # Source diversity
        source_count = sources_used.get(source, 0)
        if source_count >= 2:  # Επιτρέπουμε μέχρι 2 από ίδια πηγή
            continue
        
        # Προσθήκη στη λίστα επιλογών
        selected.append(doc)
        seen_titles.add(title_lower)
        sources_used[source] = source_count + 1
        
        print(f"Selected: {source} | {age_days}d | '{title}'")
    
    print(f" FINAL SELECTION: {len(selected)} ARTICLES from {len(sources_used)} sources")
    
    return selected

def _calculate_aggressive_recency_bonus(age_days: int) -> float:
    """
    recency bonus
    """
    if age_days <= 1:      # Σήμερα/Χθες
        return 0.8
    elif age_days <= 2:    # 2 μέρες
        return 0.6
    elif age_days <= 3:    # 3 μέρες
        return 0.4
    elif age_days <= 7:    # 1 εβδομάδα
        return 0.2
    else:
        return 0.0

def _get_article_source(doc) -> str:
    """
    Ανάκτηση πηγής με debug information
    """
    # Debug: δείξε τι metadata υπάρχει
    metadata = doc.metadata
    print(f" SOURCE DEBUG: title='{metadata.get('title', '')[:50]}...'")
    print(f"   source='{metadata.get('source', 'MISSING')}'")
    print(f"   link='{metadata.get('link', 'MISSING')}'")
    
    #  source field
    source = metadata.get("source", "").strip()
    if source and source.lower() not in ["unknown", "unknown source", "", "missing"]:
        print(f" Using source field: {source}")
        return source
    
    # : link analysis
    link = metadata.get("link", "")
    domain_source = get_source_from_link(link)
    if domain_source:
        print(f" Using link analysis: {domain_source}")
        return domain_source
    
    # category ως fallback
    category = metadata.get("category", "")
    if category and category != "General":
        print(f" Using category as source: {category}")
        return f"Category: {category}"
    
    print(f" No source found, using 'Various Sources'")
    return "Various Sources"

def _get_source_from_link(link: str) -> str:
    """
    Εξάγει πηγή από URL με ακριβή αντιστοίχιση
    """
    if not link:
        return ""
    
    link_lower = link.lower()
    
    # Ακριβής αντιστοίχιση domains
    domain_mapping = {
        "theverge.com": "The Verge",
        "techradar.com": "TechRadar", 
        "apnews.com": "Associated Press",
        "newsbeast.gr": "Newsbeast",
        "naftemporiki.gr": "Naftemporiki",
        "theguardian.com": "The Guardian",
        "guardian.com": "The Guardian",
        "techcrunch.com": "TechCrunch",
        "skai.gr": "SKAI",
        "in.gr": "In.gr",
        "tovima.gr": "To Vima",
        "documentonews.gr": "Documento",
        "greekreporter.com": "Greek Reporter",
        "abcnews.go.com": "ABC News",
        "npr.org": "NPR News",
        "eft": "Eleftheros Typos",
        "tanea.gr": "TaNea"
    }
    
    for domain, source_name in domain_mapping.items():
        if domain in link_lower:
            return source_name
    
    return ""

def generate_multi_article_answer(user_query: str, articles: list):
    """
    Δημιουργεί comprehensive απάντηση από πολλά άρθρα 
    """
    if len(articles) == 0:
        print(" No articles provided for multi-article answer")
        return None
    
    print(f" MULTI-ARTICLE ANALYSIS: {len(articles)} articles")
    
    # Support για 1 άρθρο επίσης
    if len(articles) == 1:
        print("Single article - using single article processing")
        doc = articles[0]
        title = doc.metadata.get("title", "Unknown Title")
        source = _get_article_source(doc)
        content = doc.page_content[:1200]
        
        lang = _detect_lang(user_query)
        if lang == "en":
            prompt = f"""
[SYSTEM]
Provide a comprehensive and detailed answer based on this news article.

[ARTICLE]
Title: {title}
Source: {source}
Content: {content}

[QUESTION]
{user_query}

Provide a thorough answer based on the article:
"""
        else:
            prompt = f"""
[ΣΥΣΤΗΜΑ]
Δώσε μια ολοκληρωμένη και λεπτομερή απάντηση βασισμένη σε αυτό το ειδησεογραφικό άρθρο.

[ΑΡΘΡΟ]
Τίτλος: {title}
Πηγή: {source}
Περιεχόμενο: {content}

[ΕΡΩΤΗΣΗ]
{user_query}

Δώσε μια διεξοδική απάντηση βασισμένη στο άρθρο:
"""
        
        llm = get_llm()
        if not llm:
            return None
            
        answer = generate_answer(llm, prompt)
        
        if answer and _is_meaningful_answer(answer):
            print(" SUCCESS: Generated comprehensive single-article answer")
            return answer
        else:
            print(" Single-article answer failed quality check")
            return None
    
    # Πρωτογενής λογική για 2+ άρθρα
    unique_sources = list(set([_get_article_source(doc) for doc in articles]))
    
    print(f"MULTI-ARTICLE ANALYSIS: {len(articles)} articles from {len(unique_sources)} sources: {unique_sources}")
    
    # Προετοιμασία περιεχομένου με structured format
    articles_content = []
    source_info = []
    
    for i, doc in enumerate(articles):
        title = doc.metadata.get("title", "Unknown Title")
        source = _get_article_source(doc)
        pub_date = doc.metadata.get("published_date", "Unknown Date")
        content = doc.page_content[:800]  
        
        # Extract key sentences based on query relevance
        key_sentences = _extract_key_sentences(content, user_query, max_sentences=3)
        
        articles_content.append(f"""
ARTICLE {i+1} [Source: {source}, Date: {pub_date}]:
Title: {title}
Key Information: {' '.join(key_sentences)}
""")
        
        source_info.append(f"{source} ({pub_date[:10]})")
    
    all_content = "\n".join(articles_content)
    sources_summary = ", ".join(unique_sources)
    
    # Προσδιορισμός γλώσσας
    lang = _detect_lang(user_query)
    
    if lang == "en":
        prompt = f"""
[SYSTEM]
You are a professional news analyst. Create a comprehensive answer by SYNTHESIZING information from MULTIPLE NEWS ARTICLES.

CRITICAL RULES:
- Create ONE coherent, well-structured paragraph (4-7 sentences)
- COMBINE information organically from all articles - DO NOT list them separately
- Highlight DIFFERENT aspects or developments from each article
- Focus on the most recent and important information
- Include specific dates, key facts, and developments mentioned
- Maintain neutral, factual journalistic tone throughout
- If articles provide complementary information, synthesize them naturally
- If information is conflicting, present different perspectives

[SOURCE INFORMATION]
Synthesizing information from {len(articles)} articles ({len(unique_sources)} sources): {sources_summary}

[ARTICLES CONTENT]
{all_content}

[USER QUESTION]
{user_query}

Create a comprehensive synthesized answer that addresses the user's question:
"""
    else:
        prompt = f"""
[ΣΥΣΤΗΜΑ]
Είσαι επαγγελματίας ειδησεογραφικός αναλυτής. Δημιούργησε μια ολοκληρωμένη απάντηση ΣΥΝΘΕΤΟΝΤΑΣ πληροφορίες από ΠΟΛΛΑ ΕΙΔΗΣΕΟΓΡΑΦΙΚΑ ΑΡΘΡΑ.

ΚΡΙΣΙΜΟΙ ΚΑΝΟΝΕΣ:
- Δημιούργησε ΕΝΑν συνεκτικό, καλά δομημένο παράγραφο (4-7 προτάσεις)
- ΣΥΝΔΥΑΣΕ τις πληροφορίες οργανικά από όλα τα άρθρα - ΜΗΝ τα λίστας ξεχωριστά
- Επισημάνε ΔΙΑΦΟΡΕΤΙΚΕς πτυχές ή εξελίξεις από κάθε άρθρο
- Εστίασε στις πιο πρόσφατες και σημαντικές πληροφορίες
- Συμπερίλαβε συγκεκριμένες ημερομηνίες, βασικά γεγονότα και εξελίξεις
- Διατήρησε ουδέτερο, γεγοντολογικό δημοσιογραφικό ύφος
- Αν τα άρθρα παρέχουν συμπληρωματικές πληροφορίες, σύνθεσέ τες φυσικά
- Αν υπάρχουν αντιφάσεις, παρουσίασε διαφορετικές προοπτικές

[ΠΛΗΡΟΦΟΡΙΕΣ ΠΗΓΩΝ]
Σύνθεση πληροφοριών από {len(articles)} άρθρα ({len(unique_sources)} πηγές): {sources_summary}

[ΠΕΡΙΕΧΟΜΕΝΟ ΑΡΘΡΩΝ]
{all_content}

[ΕΡΩΤΗΣΗ ΧΡΗΣΤΗ]
{user_query}

Δημιούργησε μια ολοκληρωμένη σύνθετη απάντηση που απαντά στην ερώτηση του χρήστη:
"""
    
    llm = get_llm()
    if not llm:
        return None
        
    answer = generate_answer(llm, prompt)
    
    if not answer or len(answer.strip()) < 50:
        print(" Multi-article answer too short or empty")
        return None
    
    # Προσθήκη source attribution
    if len(unique_sources) > 0:
        if lang == "en":
            answer = f"{answer}\n\n Based on reporting from: {sources_summary}"
        else:
            answer = f"{answer}\n\n Βάσει αναφορών από: {sources_summary}"
    
    print("SUCCESS: Generated comprehensive multi-article answer")
    return answer

def _extract_key_sentences(text: str, query: str, max_sentences: int = 3) -> list:
    """
    Εξάγει τις πιο σχετικές προτάσεις από το κείμενο βάσει του query
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    
    # Απλός scoring βασισμένος σε keyword matching
    query_terms = query.lower().split()
    scored_sentences = []
    
    for sentence in sentences:
        if len(sentence) < 20:  # Πολύ μικρές προτάσεις
            continue
            
        score = 0
        sentence_lower = sentence.lower()
        
        # Bonus για query terms
        for term in query_terms:
            if len(term) > 3 and term in sentence_lower:
                score += 1
        
        # Bonus για πρόσφατες πληροφορίες (dates, numbers)
        if any(indicator in sentence_lower for indicator in ['2025', '2024', 'today', 'yesterday', 'this week']):
            score += 0.5
            
        scored_sentences.append((score, sentence))
    
    # Ταξινόμηση και επιλογή
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    return [sentence for _, sentence in scored_sentences[:max_sentences]]

def handle_followup_question(session_id: str, followup_question: str, original_query: str = ""):
    context = conversation_manager.get_article_context(session_id)
    
    # Βελτιωμένη απάντηση από ίδιο άρθρο
    if context and context.get('article'):
        article_answer = _try_answer_from_same_article(context, followup_question, original_query)
        if article_answer and _is_meaningful_answer(article_answer):
            print(" Απάντηση από ίδιο άρθρο")
            return article_answer
    
    #  Ενισχυμένη αναζήτηση με περισσότερα άρθρα
    print(" Αναζήτηση σε νέα άρθρα για follow-up")
    
    search_variants = [
        followup_question,
        f"{followup_question} {original_query}",
        f"{original_query} {followup_question}",
    ]
    
    all_hits = []
    for variant in search_variants:
        print(f"Αναζήτηση: '{variant}'")
        hits = similarity_search(variant, k=15)  
        all_hits.extend(hits)
    
    unique_hits = []
    seen_content_hashes = set()
    for doc, score in all_hits:
        content_hash = hash(doc.page_content[:400])  # Περισσότερο content για καλύτερο dedup
        if content_hash not in seen_content_hashes:
            unique_hits.append((doc, score))
            seen_content_hashes.add(content_hash)
    
    print(f" Βρέθηκαν {len(unique_hits)} μοναδικά άρθρα")
    
    if not unique_hits:
        return _create_fallback_answer(followup_question)
    
    # multi-article με focus στο follow-up topic
    selected_articles = select_multiple_articles(unique_hits, followup_question, max_articles=4, max_age_days=45)
    
    if len(selected_articles) >= 1:
        print(f"Multi-article follow-up με {len(selected_articles)} άρθρα")
        multi_answer = generate_multi_article_answer(followup_question, selected_articles)
        
        if multi_answer and _is_meaningful_answer(multi_answer):
            print("Χρήση multi-article για follow-up")
            if selected_articles:
                conversation_manager.store_article_context(
                    session_id, selected_articles[0], followup_question, multi_answer
                )
            return multi_answer
    
    # single article fallback
    print("Fallback σε single article")
    selected_article = _select_best_followup_article(unique_hits, followup_question, original_query)
    
    if selected_article:
        answer = _generate_answer_from_article(selected_article, followup_question, _detect_lang(followup_question))
        conversation_manager.store_article_context(session_id, selected_article, followup_question, answer)
        return answer
    
    return _create_fallback_answer(followup_question)

def _try_answer_from_same_article(context, followup_question: str, original_query: str = ""):
    """
     Εξάγει περισσότερες πληροφορίες από το ίδιο άρθρο
    """
    article = context['article']
    lang = _detect_lang(followup_question)
    
    # Εξαγωγή των σχετικών προτάσεων από το άρθρο
    sentences = _split_sentences(article['content'])
    relevant_sentences = []
    
    # Βρες όλες τις προτάσεις που σχετίζονται με το follow-up
    followup_terms = followup_question.lower().split()
    original_terms = original_query.lower().split()
    all_terms = [term for term in followup_terms + original_terms if len(term) > 3]
    
    for sentence in sentences:
        if len(sentence) < 20:
            continue
            
        sentence_lower = sentence.lower()
        relevance_score = 0
        
        # Score based on term matches
        for term in all_terms:
            if term in sentence_lower:
                relevance_score += 2
        
        # Bonus for specific information (numbers, names, dates)
        if any(char.isdigit() for char in sentence):
            relevance_score += 1
        if any(word.istitle() for word in sentence.split() if len(word) > 3):
            relevance_score += 0.5
        
        if relevance_score > 0:
            relevant_sentences.append(sentence)
    
    # Πάρε τις πιο σχετικές προτάσεις
    relevant_sentences = relevant_sentences[:10]  
    
    if not relevant_sentences:
        return None
    
    combined_content = " ".join(relevant_sentences)
    
    if lang == "en":
        prompt = f"""
[SYSTEM]
You are discussing a specific news article. Provide a COMPREHENSIVE and DETAILED answer to the follow-up question.

IMPORTANT: 
- Use ALL relevant information from the article below
- Be thorough and include specific details, names, dates, numbers
- Don't omit any relevant information

[ARTICLE CONTEXT]
Title: {article['title']}
Original Topic: {original_query}

[RELEVANT CONTENT FROM THE ARTICLE]
{combined_content}

[FOLLOW-UP QUESTION]
{followup_question}

Provide a detailed answer with all available information:
"""
    else:
        prompt = f"""
[ΣΥΣΤΗΜΑ]
Συζητάς για ένα συγκεκριμένο ειδησεογραφικό άρθρο. Δώσε μια ΟΛΟΚΛΗΡΩΜΕΝΗ και ΛΕΠΤΟΜΕΡΗ απάντηση.

ΣΗΜΑΝΤΙΚΟ:
- Χρησιμοποίησε ΟΛΕΣ τις σχετικές πληροφορίες από το άρθρο
- Να είσαι διεξοδικός και να συμπεριλάβεις συγκεκριμένες λεπτομέρειες
- Μην παραλείψεις καμία σχετική πληροφορία

[CONTEXT ΑΡΘΡΟΥ]
Τίτλος: {article['title']}
Αρχικό Θέμα: {original_query}

[ΣΧΕΤΙΚΟ ΠΕΡΙΕΧΟΜΕΝΟ ΑΠΟ ΤΟ ΑΡΘΡΟ]
{combined_content}

[ΕΠΟΜΕΝΗ ΕΡΩΤΗΣΗ]
{followup_question}

Δώσε μια λεπτομερή απάντηση με όλες τις διαθέσιμες πληροφορίες:
"""
    
    llm = get_llm()
    if not llm:
        return None
    
    answer = generate_answer(llm, prompt)
    
    # Less strict length check for follow-ups
    if answer and len(answer.strip()) > 25:
        return answer.strip()
    
    return None

def _find_relevant_articles_for_followup(followup_question: str, original_query: str = "", context=None):
    """
    Find and answer from relevant articles when same article is insufficient
    """
    print(f" SEARCHING RELEVANT ARTICLES for: '{followup_question}'")
    
    # Search query combining follow-up and original context
    search_query = f"{followup_question} {original_query}"
    
    # Search for relevant articles
    hits_with_raw = similarity_search(search_query, k=10)
    
    if not hits_with_raw:
        return _create_fallback_answer(followup_question)
    
    # Select best article for this specific follow-up
    selected_article = _select_best_followup_article(hits_with_raw, followup_question, original_query)
    
    if not selected_article:
        return _create_fallback_answer(followup_question)
    
    # Generate answer from the new article
    lang = _detect_lang(followup_question)
    answer = _generate_answer_from_article(selected_article, followup_question, lang)
    
    # Update context with the new article for future follow-ups
    if context:
        conversation_manager.store_article_context(
            "default",  
            selected_article,
            followup_question,
            answer
        )
    
    return answer

def _select_best_followup_article(hits_with_raw, followup_question: str, original_query: str):
    """
    Καλύτερη επιλογή άρθρων για follow-up questions
    """
    current_date = datetime.now(timezone.utc)
    candidates = []
    
    for doc, raw_score in hits_with_raw:
        sim = _distance_to_similarity(raw_score)
        
        if sim < 0.05:  
            continue

        text = (doc.page_content or "").strip()
        if len(text) < 100:
            continue

        # Έλεγχος ημερομηνίας
        pub_date = doc.metadata.get("published_date")
        pub_dt = _to_utc(pub_date)
        if not pub_dt:
            continue
            
        age_days = (current_date - pub_dt).days
        
        if age_days > 90:  
            continue
        
        # Scoring για follow-up relevance
        title = doc.metadata.get("title", "").lower()
        content = text.lower()
        followup_terms = followup_question.lower().split()
        original_terms = original_query.lower().split()
        
        relevance_score = 0
        # Μεγαλύτερο bonus για follow-up terms
        for term in followup_terms:
            if len(term) > 3:
                if term in title:
                    relevance_score += 0.4  
                if term in content:
                    relevance_score += 0.15  
        
        # Bonus για original topic terms
        for term in original_terms:
            if len(term) > 3 and term in content:
                relevance_score += 0.12  
        
        # Μεγαλύτερο recency bonus
        recency_bonus = max(0, 0.4 - (age_days * 0.008))  
        
        final_score = sim + relevance_score + recency_bonus
        
        candidates.append((final_score, doc))
    
    if not candidates:
        return None
    
    # Επέστρεψε το καλύτερο candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def _generate_answer_from_article(article, question: str, lang: str):
    
    title = article.metadata.get("title", "—")
    content = (article.page_content or "").strip()[:1500]  
    
    if lang == "en":
        prompt = f"""
[SYSTEM]
Provide a comprehensive and detailed answer to the question based on the news article below.

IMPORTANT:
- Include all relevant details, names, dates, and specific information
- Be thorough and don't omit important information
- Focus on answering the specific question asked

[ARTICLE]
Title: {title}
Content: {content}

[QUESTION]
{question}

Provide a detailed, comprehensive answer:
"""
    else:
        prompt = f"""
[ΣΥΣΤΗΜΑ]
Δώσε μια ολοκληρωμένη και λεπτομερή απάντηση στην ερώτηση βασισμένος στο ειδησεογραφικό άρθρο.

ΣΗΜΑΝΤΙΚΟ:
- Συμπερίλαβε όλες τις σχετικές λεπτομέρειες, ονόματα, ημερομηνίες και συγκεκριμένες πληροφορίες
- Να είσαι διεξοδικός και μην παραλείψεις σημαντικές πληροφορίες
- Εστίασε στην απάντηση της συγκεκριμένης ερώτησης

[ΑΡΘΡΟ]
Τίτλος: {title}
Περιεχόμενο: {content}

[ΕΡΩΤΗΣΗ]
{question}

Δώσε μια λεπτομερή, ολοκληρωμένη απάντηση:
"""
    
    llm = get_llm()
    if not llm:
        return "Error: Model not available."
    
    answer = generate_answer(llm, prompt)
    
    # Less strict length requirement for follow-ups
    if answer and len(answer.strip()) > 30:
        return answer.strip()
    
    return "I couldn't generate a detailed answer from the available article."

def _is_meaningful_answer(answer: str) -> bool:
    """
    Check if the answer is meaningful - MORE RELAXED
    """
    if not answer or len(answer) < 20:  
        print(f" Answer rejected - too short: {len(answer) if answer else 0} chars")
        return False
    
    negative_phrases = [
        "i don't know", "i cannot answer", "the article doesn't", 
        "no information", "not mentioned", "not provided", "unable to",
        "there is no mention", "the article does not", "no details",
        "δεν γνωρίζω", "δεν μπορώ να απαντήσω", "το άρθρο δεν",
        "καμία πληροφορία", "δεν αναφέρεται", "αδυνατώ", "δεν υπάρχει",
        "δεν αναφέρονται", "δεν περιέχει", "κανένα στοιχείο"
    ]
    
    answer_lower = answer.lower()
    
    # Check for negative phrases 
    if any(phrase in answer_lower for phrase in negative_phrases):
        print(f"Answer rejected - contains negative phrase")
        return False
    
    # Check for substantial content indicators
    meaningful_indicators = [
        "according to", "based on", "reported", "stated", "mentioned",
        "announced", "introduced", "revealed", "unveiled", "developed",
        "launched", "released", "confirmed", "explained", "added",
        "σύμφωνα με", "βάσει", "ανέφερε", "δηλώσει", "αναφέρει",
        "ανακοίνωσε", "παρουσίασε", "αποκάλυψε", "ανέπτυξε", 
        "πρόσθεσε", "επιβεβαίωσε", "εξήγησε", "δήλωσε"
    ]
    
    # If contains meaningful indicators, it's likely good
    if any(indicator in answer_lower for indicator in meaningful_indicators):
        return True
    
    # Basic length and content check
    words = answer_lower.split()
    if len(words) >= 6:  
        return True
    
    print(f"Answer rejected - too few words: {len(words)}")
    return False

def _create_fallback_answer(question: str):
    """
     Πιο informative fallback answers
    """
    lang = _detect_lang(question)
    
    if lang == "el":
        return "Δεν βρήκα αρκετές πληροφορίες για αυτήν την ερώτηση. Μπορείς να δοκιμάσεις να ρωτήσεις κάτι πιο συγκεκριμένο ή για διαφορετικό θέμα. Επίσης, αν αυτή είναι συνέχεια προηγούμενης συζήτησης, δοκίμασε να αναφέρεις περισσότερες λεπτομέρειες."
    else:
        return "I couldn't find enough information on this specific question. You could try asking something more specific or about a different topic. If this is a follow-up to our previous discussion, try providing more context or details."

def _is_followup_question(current_query: str, original_query: str) -> bool:
    """
     Έλεγχος για follow-up
    """
    if not original_query or not current_query:
        return False
        
    current_lower = current_query.lower()
    original_lower = original_query.lower()
    
    # follow-up indicators
    strong_indicators = [
        'what about', 'and what about', 'how about', 'tell me more about',
        'τι γίνεται', 'και τι γίνεται', 'πως', 'περισσότερα για'
    ]
    
    # Αδύναμα indicators 
    weak_indicators = ['and', 'also', 'και', 'επίσης']
    
    # Έλεγχος για κοινές λέξεις μεταξύ queries
    current_words = set(current_lower.split())
    original_words = set(original_lower.split())
    common_words = current_words.intersection(original_words)
    
    # Υπολογισμός overlap
    word_overlap = len(common_words) / len(original_words) if original_words else 0
    
    # Κριτήρια για follow-up
    has_strong_indicators = any(indicator in current_lower for indicator in strong_indicators)
    has_weak_indicators = any(indicator in current_lower for indicator in weak_indicators)
    is_very_short = len(current_query.split()) <= 4
    has_high_overlap = word_overlap > 0.3  # Τουλάχιστον 30% κοινές λέξεις
    
    # Μόνο αν πληροί πολλά criteria
    if has_strong_indicators:
        return True
    elif (has_weak_indicators or is_very_short) and has_high_overlap:
        return True
    
    return False

def generate_contextual_answer(user_query, category: str = None, session_id: str = "default"):
    import re
    user_query = str(user_query).strip()

    original_query = user_query
    if _is_recency_query(user_query):
        user_query = query_with_recency(user_query)
        logger.info(f"RECENCY QUERY ENHANCED: '{original_query}' -> '{user_query}'")

    # Force multi-article για broad queries
    if _is_broad_topic_query(original_query):
        print(" BROAD TOPIC DETECTED - PRIORITIZING MULTI-ARTICLE")
        max_articles = 4
    else:
        max_articles = 3
        
    enhanced_query = query_with_recency(user_query)
    if _is_recency_query(user_query):
        print("RECENCY QUERY DETECTED")
        user_query = enhanced_query
        
    #  Έλεγχος για follow-up questions
    context = conversation_manager.get_article_context(session_id)
    if context and _is_followup_question(user_query, context.get('original_query', '')):
        print("HANDLING FOLLOW-UP QUESTION")
        return handle_followup_question(session_id, user_query, context.get('original_query', ''))
    
    context = conversation_manager.get_article_context(session_id)
    if context and _is_completely_new_topic(user_query, context.get('original_query', '')):
        print(" COMPLETELY NEW TOPIC DETECTED - Clearing previous context")
        conversation_manager.clear_context(session_id)
        context = None  # Clear the context variable too

    # Ελάχιστος έλεγχος ποιότητας ερώτησης
    if not user_query or len(user_query) < 3:
        return "Η ερώτησή σου είναι πολύ μικρή ή άδεια. Δοκίμασε να ρωτήσεις κάτι πιο συγκεκριμένο."
    if re.fullmatch(r'[\W_]+', user_query):
        return "Η ερώτησή σου περιέχει μόνο σύμβολα. Δοκίμασε ξανά με λέξεις."
    if user_query.lower() in {"hello", "hi", "hey", "γεια", "καλησπέρα"}:
        return "Γεια σου! Μπορείς να με ρωτήσεις κάτι για τις ειδήσεις, όπως \"Τι νέα για την τεχνητή νοημοσύνη;\""

    
    try:
        if vectorstore is None:
            return "Σφάλμα: Το vectorstore δεν είναι διαθέσιμο."

        # Query analysis
        query_analysis = query_router.analyze_query(user_query)
        print(f"QUERY ANALYSIS: {query_analysis}")
        
        #Generate multiple search queries
        context = conversation_manager.get_article_context(session_id)
        search_queries = query_router.generate_search_queries(user_query, query_analysis, context)
        print(f"SEARCH QUERIES: {search_queries}")
        
        # Multi-query search
        all_hits = []
        t0 = perf_counter()
        for search_query in search_queries:
            hits = similarity_search(search_query, k=20)
            all_hits.extend(hits)
        
        # Remove duplicates
        unique_hits = []
        seen_content = set()
        for doc, score in all_hits:
            content_preview = doc.page_content[:100] if doc.page_content else ""
            if content_preview not in seen_content:
                unique_hits.append((doc, score))
                seen_content.add(content_preview)
        
        hits_with_raw = unique_hits[:40]
        t_faiss = (perf_counter() - t0) * 1000.0

        # Entity extraction
        entities = entity_rag._extract_entities_from_text(user_query)  # Μόνο από το query
        print(f" EXTRACTED ENTITIES: { {k: list(v)[:3] for k, v in entities.items() if v} }")

        # DEBUG: Show results
        print("TOP 5 SEARCH RESULTS:")
        for i, (doc, score) in enumerate(hits_with_raw[:5]):
            title = doc.metadata.get('title', 'No title')[:60]
            source = _get_article_source(doc)
            print(f"  {i+1}. '{title}'... | {source} | Score: {score:.3f}")

        # DEBUG: Show top search results with dates
        print(" TOP 10 SEARCH RESULTS:")
        for i, (doc, score) in enumerate(hits_with_raw[:10]):
            title = doc.metadata.get('title', 'No title')[:70]
            date = doc.metadata.get('published_date', 'No date')
            source = doc.metadata.get('link', 'No link')[:40]
            
            # Calculate actual age
            try:
                pub_dt = _to_utc(date)
                if pub_dt:
                    now = datetime.now(timezone.utc)
                    age_days = (now - pub_dt).days
                else:
                    age_days = "N/A"
            except:
                age_days = "N/A"
                
            print(f"  {i+1}. '{title}'...")
            print(f"     Date: {date} | Age: {age_days} days | Source: {source}")
            print(f"     Score: {score:.3f}")
            print()

        if not hits_with_raw:
            return _fallback_no_context(user_query)

        # Keyword boost βάσει του query
        query_kws = _keywords_for_query(user_query)
        if query_kws:
            boosted = []
            for doc, raw in hits_with_raw:
                text = f"{doc.metadata.get('title','')} {doc.page_content or ''}".lower()
                overlap = sum(1 for k in query_kws if k in text)
                new_raw = max(0.0, float(raw) - 0.03 * overlap)
                boosted.append((doc, new_raw))
            hits_with_raw = boosted

        # Category boost βάσει category param
        if category:
            cat = category.lower().strip()
            boosted = []
            for doc, raw in hits_with_raw:
                doc_cat = (doc.metadata.get("category") or "").lower()
                bonus = 0.05 if doc_cat == cat else 0.0
                new_raw = max(0.0, float(raw) - bonus)
                boosted.append((doc, new_raw))
            hits_with_raw = boosted

        # BM25 re-rank
        if RERANK_BM25_ENABLED:
            t1 = perf_counter()
            hits_with_raw = _bm25_rerank(hits_with_raw, user_query, top_m=15)
            t_bm25 = (perf_counter() - t1) * 1000.0
        else:
            t_bm25 = 0.0

        # Multi-article answer 
        selected_articles = select_multiple_articles(hits_with_raw, user_query, max_articles=max_articles, max_age_days=30)
        multi_article_answer = None
        
        # Χρησιμοποίησε multi-article ακόμα και με 1 άρθρο
        if len(selected_articles) >= 1:  
            print(f"ATTEMPTING MULTI-ARTICLE ANSWER with {len(selected_articles)} articles")
            multi_article_answer = generate_multi_article_answer(user_query, selected_articles)
        
        # Χρησιμοποίησε multi-article αν είναι meaningful
        if multi_article_answer and _is_meaningful_answer(multi_article_answer):
            logger.info("USING MULTI-ARTICLE ANSWER")
            # Αποθήκευση context από το πρώτο άρθρο για follow-up questions
            if selected_articles:
                conversation_manager.store_article_context(session_id, selected_articles[0], user_query, multi_article_answer)
            logging.info(f"[timings] faiss_ms={t_faiss:.1f} bm25_ms={t_bm25:.1f}")
            return multi_article_answer
        
        # FALLBACK: Single article (μόνο αν multi-article απέτυχε)
        logger.info("FALLBACK TO SINGLE ARTICLE")
        chosen, topic_match = _select_single_article(
            hits_with_raw,
            query_kws=query_kws,
            query=user_query,
            hard_recency=False,
            min_sim=0.04  
        )
        
        # relaxed pass: αν δεν βρεθεί τίποτα, επέτρεψε παλιά άρθρα & χαμήλωσε λίγο το min_sim
        if not chosen:
            chosen, topic_match = _select_single_article(
                hits_with_raw,
                query_kws=query_kws,
                query=user_query,
                hard_recency=False,
                min_sim=0.02
            )
        
        # last resort: αν ακόμα δεν υπάρχει, πάρε το πρώτο hit «ως έχει»
        if not chosen and hits_with_raw:
            doc0, _raw0 = hits_with_raw[0]
            chosen, topic_match = doc0, False

        if not chosen:
            return _fallback_no_context(user_query)

        # DEBUG: Show final selected article
        if chosen:
            selected_date = chosen.metadata.get('published_date', 'No date')
            selected_title = chosen.metadata.get('title', 'No title')
            try:
                pub_dt = _to_utc(selected_date)
                if pub_dt:
                    now = datetime.now(timezone.utc)
                    age_days = (now - pub_dt).days
                else:
                    age_days = "N/A"
            except:
                age_days = "N/A"
                
            print(f" FINAL SELECTED ARTICLE:")
            print(f"   Title: {selected_title}")
            print(f"   Date: {selected_date} (Age: {age_days} days)")
            print(f"   Source: {chosen.metadata.get('link', 'No link')}")
            print()

        # Prompt για ένα άρθρο — πρώτα προσπαθούμε extractive snippets
        lang = _detect_lang(user_query)
        snippets = _extractive_snippets(chosen, user_query, k=SNIPPETS_K, min_sim=SNIPPETS_MIN_SIM) if EXTRACTIVE_ENABLED else []
        if snippets:
            prompt = _build_prompt_from_snippets(user_query, chosen, snippets, lang)
        else:
            prompt = _build_prompt_one_article(user_query, chosen, lang, topic_match)

        t2 = perf_counter()
        answer = generate_answer(get_llm(), prompt)
        t_llm = (perf_counter() - t2) * 1000.0

        # ασφάλεια τύπου
        if not isinstance(answer, str):
            answer = str(answer or "")
        answer = answer.strip()

        # fallback αν άδειο/πολύ μικρό
        if not answer or len(answer) < 10:
            return _fallback_no_context(user_query)

        # Post-processing: αφαίρεση τυχόν labels
        answer = re.sub(r'(?im)^\s*(topic\s*match\s*:.*)$', '', answer).strip()
        answer = re.sub(r'(?im)^\s*(title|date\s*\(utc\)|link|source|excerpt|άρθρο|τίτλος|ημερομηνία|σύνδεσμος|απόσπασμα)\s*:\s*.*$','',
        answer
        ).strip()
        answer = re.sub(r'\n{2,}', '\n', answer).strip()

        # Αποθήκευση context για follow-up questions
        conversation_manager.store_article_context(session_id, chosen, user_query, answer)

        logging.info(f"[timings] faiss_ms={t_faiss:.1f} bm25_ms={t_bm25:.1f} llm_ms={t_llm:.1f}")
        return answer

    except Exception as e:
        logging.exception(f"Σφάλμα κατά την επεξεργασία της ερώτησης: {str(e)}")
        return f"Σφάλμα κατά την επεξεργασία της ερώτησης: {str(e)}"
    
def query_with_recency(query: str) -> str:
    """
    Προσθέτει recency keywords στο query
    """
    query_lower = query.lower()
    
    recency_keywords = [
        "latest", "recent", "new", "today", "yesterday", "this week", 
        "current", "breaking", "just announced", "πρόσφατα", "νέα", 
        "σήμερα", "χθες", "τελευταία", "τελευταίο"
    ]
    
    # Αν το query δεν έχει ήδη recency keywords, πρόσθεσε
    if not any(keyword in query_lower for keyword in recency_keywords):
        current_year = str(datetime.now().year)
        return query + f" latest recent {current_year}"
    
    return query

def _is_recency_query(query: str) -> bool:
    """
    Ελέγχει αν το query ζητά πρόσφατες πληροφορίες
    """
    recency_indicators = [
        "latest", "recent", "new", "today", "yesterday", "this week",
        "current", "breaking", "just", "πρόσφατα", "νέα", "σήμερα",
        "χθες", "τελευταία", "τελευταίο", "πιο πρόσφατο", "πρόσφατο"
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in recency_indicators)

# πολύ απλή tokenization για en/el
_punct_re = re.compile(r"[^\wΆ-ώ]+", re.UNICODE)


def _tok(text: str):
    t = (text or "").lower()
    t = _punct_re.sub(" ", t)
    return [w for w in t.split() if len(w) > 1]

def _bm25_rerank(hits_with_raw, query: str, top_m: int = 10):
    """
    hits_with_raw: [(doc, raw_score), ...] από FAISS
    επιστρέφει ίδια δομή αλλά re-ordered από BM25 πάνω σε (title+content).
    """
    if not hits_with_raw:
        return hits_with_raw
    docs = [d for d, _ in hits_with_raw]
    corpus = [_tok(f"{d.metadata.get('title','')} {d.page_content or ''}") for d in docs]
    bm25 = BM25Okapi(corpus)
    qtok = _tok(query)
    scores = bm25.get_scores(qtok)
    ranked = sorted(zip(hits_with_raw, scores), key=lambda x: x[1], reverse=True)
    reord = [pair for (pair, _score) in ranked[:top_m]]
    return reord

# Ρύθμιση καταγραφής σφαλμάτων
logging.basicConfig(level=logging.INFO)

# 🔹 Φόρτωση vectorstore στο startup
vectorstore = load_vectorstore()

# 🔹 Lazy-loading του LLM
llm_instance = None
def get_llm():
    global llm_instance
    if llm_instance is None:
        llm_instance = load_llm()
    return llm_instance

# Topic helpers

TOPIC_KEYWORDS = {
    "ai": ["ai", "artificial intelligence", "machine learning", "neural", "open-source ai", "τεχνητή νοημοσύνη"],
    "politics": ["politics", "political", "policy", "minister", "government", "βουλή", "πολιτική", "κόμμα"],
    "sports": ["sports", "σπορ", "athletics", "match", "game", "team", "league", "πρωτάθλημα"],
    "football": ["football", "soccer", "ποδόσφ", "goal", "match", "league", "premier league", "uefa", "champions"],
    "greece": ["greece", "greek", "athens", "ελλάδα", "ελλάδα", "αθήνα", "κυβέρνηση", "ελλην"],
}

def _keywords_for_query(q: str):
    ql = (q or "").lower()
    found = []
    for _topic, kws in TOPIC_KEYWORDS.items():
        if any(k in ql for k in kws):
            found.extend(kws)
    return list(set(found))

def _text_contains_any(text: str, kws) -> bool:
    if not kws:
        return True
    tl = (text or "").lower()
    return any(k in tl for k in kws)

def filter_recent_documents(documents, days=30, desired_category=None):
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    filtered_docs = []
    for doc in documents:
        metadata = doc.metadata
        published_date_str = metadata.get("published_date")
        category = metadata.get("category", "General").lower()
        if not published_date_str:
            continue
        try:
            published_date = parser.parse(published_date_str)
            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=timezone.utc)
            if published_date >= cutoff_date:
                if desired_category:
                    if category == desired_category.lower():
                        filtered_docs.append((doc, published_date))
                else:
                    filtered_docs.append((doc, published_date))
        except Exception as e:
            logging.error(f" Σφάλμα parsing ημερομηνίας: {published_date_str} -> {e}")
            continue
    logging.info(f" Νέα φίλτρα άρθρων: {len(filtered_docs)} / {len(documents)}")
    return filtered_docs

def _is_completely_new_topic(current_query: str, previous_query: str) -> bool:
    """
    Ελέγχει αν η νέα ερώτηση είναι εντελώς διαφορετικό θέμα
    """
    if not previous_query:
        return True
        
    current_lower = current_query.lower()
    previous_lower = previous_query.lower()
    
    # Keywords για διαφορετικά θέματα
    topic_keywords = {
        'politics': ['trump', 'biden', 'election', 'government', 'senate', 'congress'],
        'technology': ['ai', 'artificial intelligence', 'tech', 'computer', 'software'],
        'sports': ['football', 'basketball', 'team', 'game', 'league'],
        'economy': ['economy', 'market', 'stock', 'inflation', 'money'],
        'weather': ['weather', 'temperature', 'snow', 'rain', 'cold']
    }
    
    # Βρες το topic της κάθε ερώτησης
    current_topic = None
    previous_topic = None
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in current_lower for keyword in keywords):
            current_topic = topic
        if any(keyword in previous_lower for keyword in keywords):
            previous_topic = topic
    
    # Αν είναι διαφορετικά topics, είναι νέα συζήτηση
    return current_topic != previous_topic and current_topic is not None and previous_topic is not None

# Scoring params & helpers 
MAX_AGE_DAYS = 30        
RECENCY_WEIGHT = 0.5 
TAU_DAYS = 3.0         
MIN_SIM = 18.0         
MIN_LEN = 300           

def _to_utc(dt_str):
    """Μετατρέπει οποιοδήποτε date string σε aware UTC datetime."""
    if not dt_str:
        return None
    dt = parser.parse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _age_days(dt_str):
    dt = _to_utc(dt_str)
    if not dt:
        return 10**9
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)

def _recency_boost(dt_str):
    age = _age_days(dt_str)
   
    if age > 40:  
        return 0.0
   
    if age <= 1:    # Σήμερα
        return 0.8  # Από 1.0
    elif age <= 3:  # 1-3 ημέρες
        return 0.6  # Από 0.9
    elif age <= 7:  # 1 εβδομάδα
        return 0.4  # Από 0.7
    elif age <= 14: # 2 εβδομάδες
        return 0.2  # Από 0.4
    elif age <= 30: # 1 μήνας
        return 0.1  # Νέο
    else:           # 1-2 μήνες
        return 0.05 # Νέο

def _distance_to_similarity(raw_score: float) -> float:
    """
    Μετατροπή FAISS distance -> pseudo-similarity ∈ [0,1].
    """
    try:
        d = float(raw_score)
    except Exception:
        d = 1.0
    return 1.0 / (1.0 + d) if d > 1.0 else max(0.0, 1.0 - d)

def _final_score(similarity: float, published_iso: str, query: str = "") -> float:
    r = _recency_boost(published_iso)
    
    # DYNAMIC WEIGHTING based on query
    query_lower = (query or "").lower()
    
    # Αν η ερώτηση ζητάει ρητά πρόσφατα νέα
    if any(keyword in query_lower for keyword in 
           ["today", "latest", "recent", "breaking", "just", "new", "πρόσφατα", "σήμερα", "νέα"]):
        recency_weight = 0.65  #  Υψηλότερη προτεραιότητα σε recency
    else:
        recency_weight = RECENCY_WEIGHT  # Normal weight
    
    return similarity * (1.0 - recency_weight) + recency_weight * r

def _select_single_article(hits_with_raw, query_kws=None, query: str = "", *, hard_recency=False, min_sim=0.04):  # Μείωση από 0.06 σε 0.04
    """
     single article selection with similarity threshold
    """
    query_kws = query_kws or []
    candidates = []
    
    current_date = datetime.now(timezone.utc)
    
    print(f"SINGLE ARTICLE SELECTION")
    print(f"Processing {len(hits_with_raw)} hits | hard_recency: {hard_recency} | min_sim: {min_sim}")

    for i, (doc, raw) in enumerate(hits_with_raw):
        sim = _distance_to_similarity(raw)
        
        if sim < min_sim:
            continue

        text = (doc.page_content or "").strip()
        if len(text) < 80:  
            continue

        pub = doc.metadata.get("published_date")
        pub_dt = _to_utc(pub)
        
        if not pub_dt:
            continue
            
        # Recency calculation
        age_days = (current_date - pub_dt).days
        
        # Πιο χαλαρό recency filtering
        if hard_recency and age_days > 45: 
            continue

        # Topic matching
        topic_match = _text_contains_any(f"{doc.metadata.get('title','')} {text}", query_kws)
    
        # Enhanced scoring
        base_score = _final_score(sim, pub, query)
        
        # Μέτριο bonus για topic match
        if topic_match and query_kws:
            base_score += 0.10  
            
        # Μέτριο bonus για πρόσφατα άρθρα
        if age_days <= 7:
            base_score += 0.12  
        elif age_days <= 14:
            base_score += 0.06
            
        # Πολύ μικρό penalty για παλιά άρθρα
        if age_days > 90:  
            base_score *= 0.85
            
        candidates.append((base_score, doc, sim, pub, topic_match, age_days))

    if not candidates:
        print("No candidates passed filtering - using fallback")
        # Fallback: πάρε το κορυφαίο από similarity search με πολύ χαμηλό threshold
        for doc, raw in hits_with_raw[:3]:
            if _distance_to_similarity(raw) > 0.03:  # Πολύ χαμηλό threshold για fallback
                pub_date = doc.metadata.get("published_date", "No date")
                title = doc.metadata.get('title', 'No title')[:60]
                print(f" Using result with low similarity: {pub_date} | '{title}...'")
                return doc, False
        return None, False

    # Ταξινόμηση και επιλογή
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    print(" TOP CANDIDATES:")
    for i, (score, doc, sim, pub, topic_match, age) in enumerate(candidates[:5]):
        title = doc.metadata.get('title', 'No title')
        source = _get_article_source(doc)
        print(f"  {i+1}. Score: {score:.3f} | Age: {age}d | Sim: {sim:.3f} | {source} | '{title[:50]}...'")

    best_score, best_doc, best_sim, best_pub, best_match, best_age = candidates[0]
    
    print(f"SELECTED: '{best_doc.metadata.get('title','—')}' | Score: {best_score:.3f} | Age: {best_age}d")
    return best_doc, best_match

# Extractive snippets helpers 
_embedder = None
def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return _embedder

# απλός splitter που δουλεύει en/el (., ?, !, ελληνική άνω τελεία '·', ellipsis …)
_SENT_SPLIT_RE = re2.compile(r'(?<=[\.\?\!·…])\s+(?=[A-ZΑ-ΩΉΊΌΎΏΆΈΉΊΌΎΏ])')

def _split_sentences(text: str):
    txt = (text or "").strip()
    parts = _SENT_SPLIT_RE.split(txt) if txt else []
    # πέτα πολύ κοντές/σκουπιδένιες προτάσεις και κόψε υπερβολικό μήκος
    return [s.strip() for s in parts if len(s.strip()) >= 40][:60]

def _cos(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

def _mmr_select(sentences, query_vec, sent_vecs, k=4, lambda_param=0.7):
    if not sentences:
        return [], []
    sims = np.array([_cos(v, query_vec) for v in sent_vecs])
    selected = []
    candidates = list(range(len(sentences)))
    # πρώτο: max relevance
    first = int(np.argmax(sims))
    selected.append(first)
    candidates.remove(first)
    while len(selected) < min(k, len(sentences)):
        mmr_scores = []
        for i in candidates:
            rel = sims[i]
            div = max(_cos(sent_vecs[i], sent_vecs[j]) for j in selected)
            mmr = lambda_param * rel - (1.0 - lambda_param) * div
            mmr_scores.append((mmr, i))
        mmr_scores.sort(reverse=True)
        best_i = mmr_scores[0][1]
        selected.append(best_i)
        candidates.remove(best_i)
    return [sentences[i] for i in selected], [sims[i] for i in selected]

def _extractive_snippets(doc, user_query, k=4, min_sim=0.20):
    text = (doc.page_content or "").strip()
    sents = _split_sentences(text)
    if not sents:
        return []
    emb = _get_embedder()
    try:
        sent_vecs = np.array(emb.embed_documents(sents))
        query_vec = np.array(emb.embed_query(user_query))
    except Exception as _e:
        logging.warning(f"Embed error (snippets): {_e}")
        return []
    chosen, sims = _mmr_select(sents, query_vec, sent_vecs, k=k, lambda_param=0.7)
    if chosen and float(np.mean(sims)) >= min_sim:
        return chosen
    return []

def query_for_search(query: str) -> str:
    """
    Βελτιώνει τα queries για καλύτερη αναζήτηση
    """
    query_lower = query.lower()
    
    # Συγκεκριμένα keywords
    if any(word in query_lower for word in ["latest", "recent", "new", "today", "breaking"]):
        return query + " news updates 2025 current"
    
    # Γενική βελτίωση
    return query

def _build_prompt_one_article(user_query: str, doc, lang: str, topic_match: bool) -> str:
    title = doc.metadata.get("title", "—")
    link = doc.metadata.get("link", "")
    pub_iso_dt = _to_utc(doc.metadata.get("published_date"))
    pub_iso = pub_iso_dt.isoformat() if pub_iso_dt else "Άγνωστη ημερομηνία"
    content = (doc.page_content or "").strip()[:1200]

    if lang == "en":
        return f"""
[SYSTEM]
You are a news assistant. Answer STRICTLY from the SINGLE article text below.

OUTPUT RULES (MUST FOLLOW EXACTLY):
- Write a single paragraph with 3–5 sentences. No headings, no lists, no labels.
- First sentence MUST begin with: "Based on the {pub_iso} article:"
- Use ONLY the article. Do NOT invent facts. Do NOT ask the user questions.
- Mention the UTC date only in the first sentence. Include the source link only ONCE as the LAST token: {link}
- Do NOT mention internal flags or sections. Do NOT write the words "Topic match", "Title", "Source", "Excerpt", or similar.
- If and only if the article does NOT cover the user's topic directly, add one short note as the LAST sentence before the link: "Note: the provided article does not cover the requested topic directly."
- Do NOT produce lists or bullet points under any circumstance.

[ARTICLE TEXT]
Title: {title}
Date (UTC): {pub_iso}
Link: {link}
Excerpt:
\"\"\"
{content}
\"\"\"


[USER QUESTION]
{user_query}

[INTERNAL FLAG — DO NOT MENTION IN THE ANSWER]
topic_match={str(topic_match).lower()}
""".strip()

    # default: Greek
    return f"""
[ΣΥΣΤΗΜΑ]
Είσαι βοηθός ειδήσεων. Απάντησε ΑΠΟΚΛΕΙΣΤΙΚΑ από το παρακάτω ΕΝΑ άρθρο.

ΚΑΝΟΝΕΣ ΕΞΟΔΟΥ (ΑΚΡΙΒΩΣ ΕΤΣΙ):
- Γράψε ένα ενιαίο παράγραφο με 3–5 προτάσεις. Χωρίς επικεφαλίδες, χωρίς λίστες, χωρίς labels.
- Η πρώτη πρόταση ΠΡΕΠΕΙ να ξεκινά με: "Βάσει άρθρου της {pub_iso}:"
- Χρησιμοποίησε ΜΟΝΟ το άρθρο. ΜΗΝ εφευρίσκεις στοιχεία. ΜΗΝ ρωτάς τον χρήστη.
- Ανάφερε την ημερομηνία UTC μόνο στην πρώτη πρόταση. Βάλε το link ΜΟΝΟ ΜΙΑ φορά ως ΤΕΛΕΥΤΑΙΟ token: {link}
- ΜΗΝ αναφέρεις εσωτερικές σημαίες/sections. ΜΗΝ γράψεις λέξεις όπως "Σημαία", "Topic match", "Τίτλος", "Πηγή", "Απόσπασμα".
- ΜΟΝΟ αν το άρθρο δεν ταιριάζει άμεσα το ζητούμενο θέμα, πρόσθεσε ΜΙΑ σύντομη τελική πρόταση πριν το link: "Σημείωση: το άρθρο δεν καλύπτει άμεσα το ζητούμενο θέμα."
- ΜΗΝ χρησιμοποιείς λίστες ή bullets για κανέναν λόγο.

[ΚΕΙΜΕΝΟ ΑΡΘΡΟΥ]
Τίτλος: {title}
Ημερομηνία (UTC): {pub_iso}
Σύνδεσμος: {link}
Απόσπασμα:
\"\"\"
{content}
\"\"\"


[ΕΡΩΤΗΣΗ ΧΡΗΣΤΗ]
{user_query}

[ΕΣΩΤΕΡΙΚΗ ΣΗΜΑΙΑ — ΜΗΝ ΤΗΝ ΑΝΑΦΕΡΕΙΣ ΣΤΗΝ ΑΠΑΝΤΗΣΗ]
topic_match={str(topic_match).lower()}
""".strip()

def _build_prompt_from_snippets(user_query: str, doc, snippets: list[str], lang: str) -> str:
    title = doc.metadata.get("title", "—")
    link = doc.metadata.get("link", "")
    pub_iso_dt = _to_utc(doc.metadata.get("published_date"))
    pub_iso = pub_iso_dt.isoformat() if pub_iso_dt else "Άγνωστη ημερομηνία"
    joined = " ".join(snippets)[:1200]

    if lang == "en":
        return f"""
[SYSTEM]
Rewrite the provided sentences into a concise, neutral-tone news answer. No emojis, no exclamation tone.

RULES:
- ONE paragraph, 3–5 sentences.
- First sentence MUST begin: "Based on the {pub_iso} article:"
- Use ONLY the sentences below (no outside facts).
- Mention the date only in the first sentence.
- End with the source link as the LAST token: {link}
- Do NOT add disclaimers unless the sentences are irrelevant to the question.
- Do NOT produce lists or bullet points under any circumstance.

[SENTENCES]
\"\"\"
{joined}
\"\"\"


[USER QUESTION]
{user_query}
""".strip()

    return f"""
[ΣΥΣΤΗΜΑ]
Ξαναγράψε τις παρεχόμενες προτάσεις σε συνοπτική, ουδέτερου ύφους ειδησεογραφική απάντηση. Χωρίς emojis.

ΚΑΝΟΝΕΣ:
- Ένας παράγραφος, 3–5 προτάσεις.
- Η πρώτη ΠΡΕΠΕΙ να αρχίζει: "Βάσει άρθρου της {pub_iso}:"
- Χρησιμοποίησε ΜΟΝΟ τις προτάσεις (καμία εξωτερική πληροφορία).
- Ανάφερε την ημερομηνία μόνο στην πρώτη πρόταση.
- Κλείσε με το link ως ΤΕΛΕΥΤΑΙΟ token: {link}
- ΜΗΝ προσθέτεις disclaimers εκτός αν οι προτάσεις είναι άσχετες με την ερώτηση.
- ΜΗΝ χρησιμοποιείς λίστες ή bullets για κανέναν λόγο.

[ΠΡΟΤΑΣΕΙΣ]
\"\"\"
{joined}
\"\"\"


[ΕΡΩΤΗΣΗ]
{user_query}
""".strip()

def _detect_lang(q: str) -> str:
    # απλή ευρετική: αν έχει ελληνικούς χαρακτήρες -> el, αλλιώς en
    return "el" if re.search(r"[Α-Ωα-ω]", q or "") else "en"

def _fallback_no_context(q: str) -> str:
    return ("No sufficiently recent/relevant article (within 30 days). "
            "Try a more specific query or another topic.") if _detect_lang(q) == "en" else (
            "Δεν βρέθηκε αρκετά πρόσφατο/σχετικό άρθρο (εντός 30 ημερών). "
            "Δοκίμασε πιο συγκεκριμένη αναζήτηση ή άλλο θέμα.")

def _is_broad_topic_query(query: str) -> bool:
    """
    Εντοπίζει ευρείες ερωτήσεις που χρειάζονται multi-article απάντηση
    """
    broad_indicators = [
        "news about", "latest on", "developments in", "what's happening in",
        "tell me about", "what are", "how is", "current state of",
        "overview of", "trends in", "updates on", "νεα για", "εξελιξεις",
        "τι νεα", "προσφατα", "ενημερωσε με"
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in broad_indicators)
    
class EntityAwareRAG:
    """Βελτιωμένο RAG system με entity recognition και tracking"""
    
    def __init__(self):
        self.entity_cache = {}
        self.political_knowledge_base = self._initialize_political_kb()
    
    def _initialize_political_kb(self):
        """Βασική γνώση για πολιτικά συστήματα και οντοτήες"""
        return {
            'us_politics': {
                'parties': {
                    'democrat': ['democratic', 'democrat', 'dems'],
                    'republican': ['republican', 'gop', 'conservative', 'republicans']
                }
            },
            'known_relationships': {
                'trump': {'party': 'republican', 'role': 'former president'},
                'biden': {'party': 'democrat', 'role': 'president'},
                'harris': {'party': 'democrat', 'role': 'vice president'}
            }
        }
    
    def extract_and_track_entities(self, query: str, articles: list):
        """Εξαγωγή και παρακολούθηση οντοτήων από query και άρθρα"""
        entities = {
            'people': set(), 'organizations': set(), 'locations': set(),
            'dates': set(), 'political_parties': set(), 'elections': set(), 'topics': set()
        }
        
        # Εξαγωγή από query
        query_entities = self._extract_entities_from_text(query)
        entities = self._merge_entities(entities, query_entities)
        
        # Εξαγωγή από άρθρα
        article_entities = self._extract_entities_from_articles(articles)
        entities = self._merge_entities(entities, article_entities)
        
        return entities
    
    def _extract_entities_from_text(self, text: str):
        """Εξαγωγή οντοτήων από κείμενο"""
        entities = {
            'people': set(), 'organizations': set(), 'political_parties': set(),
            'locations': set(), 'dates': set(), 'elections': set(), 'topics': set()
        }
        
        if not text:
            return entities
        
        text_lower = text.lower()
        
        #  patterns για όλα τα θέματα
        patterns = {
            'people': [
                r'\b(Trump|Biden|Harris|Putin|Netanyahu|Μητσοτάκης|Τσίπρας)\b',
                r'\b(Leto|Μπασινάς|Γκαρσία|Ιβάν|Μαρτίνς)\b',
                r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'  # Γενικό pattern ονομάτων
            ],
            'organizations': [
                r'\b(White House|Congress|Senate|NATO|EU|UN)\b',
                r'\b(Champions League|Conference League|Europa League|UEFA)\b',
                r'\b(Ολυμπιακός|Παναθηναϊκός|ΑΕΚ|ΠΑΟΚ|Άρης)\b',
                r'\b(Αρχηγείο|Κοινοβούλιο|Βουλή|Υπουργείο)\b'
            ],
            'political_parties': [
                r'\b(Democrat|Republican|GOP|Conservative)\b',
                r'\b(Δημοκρατικ|Ρεπουμπλικ|Συνασπισμ|Νέα Δημοκρατία)\w*\b'
            ],
            'locations': [
                r'\b(Washington|New York|California|Texas|Athens|Thessaloniki)\b'
            ]
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entity = match.group().strip()
                        if len(entity) > 2:
                            entities[entity_type].add(entity)
                except:
                    continue
        
        # Topic detection
        topic_keywords = {
            'sports': ['football', 'soccer', 'basketball', 'team', 'match', 'league', 
                      'ποδόσφαιρο', 'μπάσκετ', 'ομάδα', 'αγώνας', 'πρωτάθλημα'],
            'politics': ['election', 'government', 'minister', 'president', 
                        'εκλογές', 'κυβέρνηση', 'υπουργός', 'πρόεδρος'],
            'economy': ['inflation', 'economy', 'tax', 'budget', 'market',
                       'πληθωρισμός', 'οικονομία', 'φόρος', 'προϋπολογισμός'],
            'technology': ['ai', 'artificial intelligence', 'tech', 'software',
                          'τεχνητή νοημοσύνη', 'τεχνολογία', 'λογισμικό']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                entities['topics'].add(topic)
        
        return entities
    
    def _extract_entities_from_articles(self, articles: list):
        """Εξαγωγή οντοτήων από άρθρα"""
        all_entities = {
            'people': set(), 'organizations': set(), 'political_parties': set(),
            'locations': set(), 'dates': set(), 'elections': set(), 'topics': set()
        }
        
        for article in articles:
            try:
                content = f"{article.metadata.get('title', '')} {article.page_content}"
                article_entities = self._extract_entities_from_text(content)
                all_entities = self._merge_entities(all_entities, article_entities)
            except:
                continue
        
        return all_entities
    
    def _merge_entities(self, entities1, entities2):
        """Συγχώνευση entities"""
        merged = {}
        for key in entities1.keys():
            merged[key] = entities1[key].union(entities2.get(key, set()))
        return merged

class QueryRouter:
    """Σύστημα ανάλυσης και routing ερωτήσεων"""
    
    def __init__(self):
        self.query_types = {
            'factual': ['who is', 'what is', 'when did', 'where is', 'which', 'ποιος είναι', 'τι είναι'],
            'analytical': ['why', 'how', 'analyze', 'explain', 'γιατί', 'πώς', 'εξήγησε'],
            'comparative': ['compare', 'difference between', 'vs', 'versus', 'σύγκρινε', 'διαφορά'],
            'followup': ['what about', 'and', 'also', 'more about', 'τι γίνεται', 'και', 'επίσης'],
            'temporal': ['latest', 'recent', 'new', 'today', 'yesterday', 'πρόσφατα', 'νέα', 'σήμερα']
        }
    
    def analyze_query(self, query: str, context: dict = None):
        """Ανάλυση ερώτησης"""
        analysis = {
            'type': 'factual',
            'intent': 'general_info',
            'requires_context': False,
            'search_strategy': 'standard',
            'language': 'en'
        }
        
        query_lower = query.lower()
        
        # Γλώσσα
        analysis['language'] = 'el' if re.search(r'[Α-Ωα-ω]', query) else 'en'
        
        # Τύπος ερώτησης
        for q_type, patterns in self.query_types.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis['type'] = q_type
                break
        
        # Intent detection
        if any(word in query_lower for word in ['who', 'person', 'candidate', 'ποιος']):
            analysis['intent'] = 'find_person'
        elif any(word in query_lower for word in ['latest', 'recent', 'new', 'πρόσφατα']):
            analysis['intent'] = 'get_latest'
        
        # Follow-up detection
        if context and self._is_followup_query(query, context):
            analysis['type'] = 'followup'
            analysis['requires_context'] = True
            analysis['search_strategy'] = 'context_aware'
        
        return analysis
    
    def _is_followup_query(self, query: str, context: dict) -> bool:
        """Έλεγχος για follow-up"""
        if not context or 'last_query' not in context:
            return False
        
        followup_indicators = ['what about', 'and', 'also', 'more about', 'τι γίνεται', 'και', 'επίσης']
        query_lower = query.lower()
        has_indicators = any(indicator in query_lower for indicator in followup_indicators)
        is_short = len(query.split()) <= 6
        
        return has_indicators or is_short
    
    def generate_search_queries(self, query: str, analysis: dict, context: dict = None):
        """Δημιουργία search queries"""
        base_queries = [query]
        
        # Queries based on analysis
        if analysis['intent'] == 'find_person':
            base_queries.extend([f"{query} news", f"{query} latest"])
        elif analysis['intent'] == 'get_latest':
            current_year = str(datetime.now().year)
            base_queries.extend([f"{query} {current_year}", f"{query} today"])
        elif analysis['type'] == 'followup' and context:
            if 'original_query' in context:
                base_queries.append(f"{query} {context['original_query']}")
        
        return base_queries

# Initialize components
entity_rag = EntityAwareRAG()
query_router = QueryRouter()