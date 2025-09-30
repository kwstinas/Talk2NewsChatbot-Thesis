const { useEffect, useRef, useState } = React;

function Message({ role, text }) {
  return (
    <div className={`msg ${role === "user" ? "user" : "bot"}`}>
      {text}
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
        <Message key={i} role={m.role} text={m.text} />
      ))}
    </div>
  );
}

function Footer({ onSend, loading }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  const handleSend = async () => {
    const v = value.trim();
    if (!v) return;
    await onSend(v);
    setValue("");               // καθαρισμός input
    if (textareaRef.current) {
      textareaRef.current.value = "";  // force reset
    }
  };

  const onKeyDown = (e) => {
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

function ChatApp() {
  const [messages, setMessages] = useState(() => {
    try {
      const raw = localStorage.getItem("t2n_chat");
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    localStorage.setItem("t2n_chat", JSON.stringify(messages));
  }, [messages]);

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
      setMessages((prev) => [...prev, { role: "assistant", text: answer }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Error: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <div className="brand">
          <img
            src="./assets/Talk2News.png"
            alt="Chatbot Logo"
            className="brand-logo"
          />
          <h1>Talk2News Chatbot</h1>
        </div>
      </div>

      <div className="card chat">
        <div className="main">
          <Messages items={messages} />
          <Footer onSend={askAPI} loading={loading} />
          <div className="meta">
            Tip: Ask specific topics (e.g., “Israel war”, “Greek politics”, “Eurobasket 2025”).
          </div>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<ChatApp />);
