const { useEffect, useRef, useState } = React;

/* ========== Message bubbles ========== */
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

  // Daily Digest:
  // A) backend endpoint   /api/digest
  // B) fallback χωρίς backend
  const requestDigest = async () => {
    setLoading(true);
    try {
      // Προσπάθησε backend-first
      const res = await fetch("/api/digest");
      if (res.ok) {
        const data = await res.json();
        const answer = (data.answer || data.result || data.text || "").trim();
        if (answer) {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", text: answer, onFav: () => addFavorite(answer) },
          ]);
          setLoading(false);
          return;
        }
      }
      // Fallback: special prompt
      await askAPI(
        "Give me a concise daily digest of today’s top 4–5 news items strictly from the crawled articles of the last 48 hours. Provide one short paragraph."
      );
    } catch {
      await askAPI(
        "Give me a concise daily digest of today’s top 4–5 news items strictly from the crawled articles of the last 48 hours. Provide one short paragraph."
      );
    } finally {
      setLoading(false);
    }
  };

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));
  const fetchLatestNewsSum = async () => {
    try {
      const res = await fetch("/api/digest?n=5&hours=72");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const text = (data.digest || "").trim() || "No digest.";
      setMessages((prev) => [...prev, { role: "assistant", text }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", text: `Error: ${err.message}` }]);
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
          <button className="btn-ghost" onClick={requestDigest} title="Daily Digest">🗞️ Latest News Sum</button>
          <button className="btn-ghost" onClick={clearChat} title="Clear chat">🧹 Clear</button>
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
