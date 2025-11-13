const { useEffect, useRef, useState } = React;

/* Message bubbles */
function Message({ role, text, onFav, isFavoritable }) {
  return (
    <div className={`msg ${role === "user" ? "user" : "bot"}`}>
      <div>{text}</div>
      {role === "assistant" && isFavoritable && (
        <div className="msg-actions">
          <button className="btn-ghost" onClick={onFav} title="Save to Favorites">⭐ Save</button>
        </div>
      )}
    </div>
  );
}

function Messages({ items }) {
  const listRef = useRef(null);
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [items]);
  return (
    <div className="messages" ref={listRef}>
      {items.map((m, i) => (
        <Message
          key={i}
          role={m.role}
          text={m.text}
          isFavoritable={true}
          onFav={m.role === "assistant" ? m.onFav : undefined}
        />
      ))}
    </div>
  );
}

/*  Footer composer  */
function Footer({ onSend, loading }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  const handleSend = async () => {
    const v = value.trim();
    if (!v) return;
    await onSend(v);
    setValue("");               
    textareaRef.current?.focus();
  };

  const onKeyDown = (e) => {
    // Enter => send (Shift+Enter για newline)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="footer">
      <textarea
        ref={textareaRef}
        className="input"
        placeholder="Ask about the news… (Enter to send, Shift+Enter for newline)"
        rows={1}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
      />
      <button className="button" onClick={handleSend} disabled={loading}>
        {loading ? "Thinking…" : "Send"}
      </button>
    </div>
  );
}

/*  Simple cards for Favorites  */
function FavoriteItem({ item, onRemove }) {
  return (
    <div className="fav-item">
      <div className="fav-text">{item.text}</div>
      <button className="btn-ghost danger" onClick={onRemove} title="Remove">✕</button>
    </div>
  );
}

/* Main app */
function ChatApp() {
  // messages
  const [messages, setMessages] = useState(() => {
    try {
      const raw = localStorage.getItem("t2n_chat");
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  });
  const [loading, setLoading] = useState(false);

  // favorites
  const [favorites, setFavorites] = useState(() => {
    try {
      const raw = localStorage.getItem("t2n_favs");
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  });

  // theme
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("t2n_theme") || "dark";
    } catch { return "dark"; }
  });

  // persist state
  useEffect(() => localStorage.setItem("t2n_chat", JSON.stringify(messages)), [messages]);
  useEffect(() => localStorage.setItem("t2n_favs", JSON.stringify(favorites)), [favorites]);
  useEffect(() => {
    localStorage.setItem("t2n_theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  // API helpers
  const askAPI = async (question) => {
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", text: question }]);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const answer = (data.answer || data.result || "").trim() || "No answer.";
      // pass an onFav handler per message so the star works
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: answer, onFav: () => addFavorite(answer) },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Error: ${err.message}`, onFav: () => addFavorite(`Error: ${err.message}`) },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const addFavorite = (text) => {
    const item = { id: `${Date.now()}_${Math.random().toString(16).slice(2)}`, text };
    setFavorites((prev) => [item, ...prev].slice(0, 50)); // keep last 50
  };

  const removeFavorite = (id) => {
    setFavorites((prev) => prev.filter((f) => f.id !== id));
  };

  const clearChat = () => {
    setMessages([]);
    try { localStorage.removeItem("t2n_chat"); } catch {}
  };

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  // Daily Digest function - COMPLETELY SEPARATE from askAPI
  const requestDigest = async () => {
    console.log("DIGEST-ONLY FLOW STARTED");
    setLoading(true);
    
    try {
      console.log("Step 1: Calling /api/digest...");
      
      const response = await fetch("/api/digest?n=5&hours=24&lang=en");
      console.log("Digest response status:", response.status);
      
      if (!response.ok) {
        console.log("Digest failed with status:", response.status);
        // Add error message
        setMessages((prev) => [
          ...prev,
          { 
            role: "assistant", 
            text: "Δεν μπόρεσα να βρω πρόσφατες ειδήσεις για σύνοψη. Δοκίμασε αργότερα.",
            onFav: () => addFavorite("Δεν βρέθηκαν πρόσφατες ειδήσεις")
          },
        ]);
        return;
      }
      
      const data = await response.json();
      console.log("Digest received:", data.digest?.length, "characters");
      
      // Add ONLY the digest response - NO user message
      setMessages((prev) => [
        ...prev,
        { 
          role: "assistant", 
          text: data.digest, 
          onFav: () => addFavorite(data.digest) 
        },
      ]);
      
      console.log("Digest added successfully - PROCESS COMPLETE");
      
    } catch (error) {
      console.error("Digest error:", error);
      // Add error message
      setMessages((prev) => [
        ...prev,
        { 
          role: "assistant", 
          text: "Σφάλμα στην ανάκτηση των ειδήσεων. Δοκίμασε αργότερα.",
          onFav: () => addFavorite("Σφάλμα ανάκτησης ειδήσεων")
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      {/* Header / Toolbar */}
      <div className="topbar">
        <div className="topbar-left">
          <img src="./assets/Talk2News.png" alt="Chatbot Logo" className="logo-lg" />
          <div className="title">Talk2News Chatbot</div>
        </div>
        <div className="topbar-actions">
          {/* ONLY ONE DIGEST BUTTON */}
          <button className="btn-ghost" onClick={requestDigest} title="Daily Digest">
            🗞️ Daily News Sum
          </button>
          <button className="btn-ghost" onClick={clearChat} title="Clear chat">
            🧹 Clear
          </button>
          <button className="btn-ghost" onClick={toggleTheme} title="Dark / Light">
            {theme === "dark" ? "🌙 Dark" : "☀️ Light"}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="card chat">
        {/* Left panel: Favorites */}
        <div className="sidebar">
          <h3>Favorites</h3>
          {favorites.length === 0 && (
            <div className="hint">No favorites yet. Click ⭐ on any answer to save it.</div>
          )}
          {favorites.map((f) => (
            <FavoriteItem key={f.id} item={f} onRemove={() => removeFavorite(f.id)} />
          ))}
        </div>

        {/* Main chat */}
        <div className="main">
          <Messages items={messages} />
          <Footer onSend={askAPI} loading={loading} />
          <div className="meta">
            Tip: Press <b>Enter</b> to send (Shift+Enter for newline). Your messages & favorites persist locally.
          </div>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<ChatApp />);