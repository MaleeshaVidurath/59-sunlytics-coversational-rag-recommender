import { useState, useEffect, useRef } from "react";
import { useSelector, useDispatch } from "react-redux";

import ChatTemplate from "../components/templates/ChatTemplate";
import Sidebar from "../components/organisms/Sidebar";
import ChatHeader from "../components/organisms/ChatHeader";
import MessageList from "../components/organisms/MessageList";
import MessageInput from "../components/molecules/MessageInput";

import { selectUser, logout } from "../store/slices/authSlice";
import { selectModel, modelReset } from "../store/slices/modelSlice";
import {
  selectSessions, selectActiveSession, selectSessionsError,
  fetchSessions, removeSession, sessionsErrorCleared,
} from "../store/slices/sessionsSlice";
import {
  selectMessages, selectSending, selectAwaitingConsent, selectChatError,
  openSession, startNewChat, sendChatMessage,
  acceptConsent, declineConsent, rateMessage, chatErrorCleared,
} from "../store/slices/chatSlice";
import { validateMessage } from "../utils/validation";

export default function ChatPage() {
  const dispatch = useDispatch();

  const user          = useSelector(selectUser);
  const model         = useSelector(selectModel);
  const sessions      = useSelector(selectSessions);
  const activeSession = useSelector(selectActiveSession);
  const messages      = useSelector(selectMessages);
  const sending       = useSelector(selectSending);
  const awaitingConsent = useSelector(selectAwaitingConsent);
  const chatError     = useSelector(selectChatError);
  const sessionsError = useSelector(selectSessionsError);

  // View-only state. The draft changes on every keystroke and the sidebar
  // toggle is a per-viewer preference — neither is worth a store round-trip,
  // and putting the draft in the store would re-render the transcript on every
  // character typed.
  const [input, setInput]         = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  // Identity comes from the session cookie, so this only needs to run once
  // per signed-in user.
  useEffect(() => { dispatch(fetchSessions()); }, [dispatch]);

  const refocus = () => inputRef.current?.focus();

  async function newChat() {
    await dispatch(startNewChat());
    // Back to model selection — the next chat picks its model afresh.
    dispatch(modelReset());
  }

  async function handleDeleteSession(sessionId) {
    if (!window.confirm("Delete this chat? All data will be removed.")) return;
    dispatch(removeSession({ sessionId }));
  }

  async function send() {
    // Validate before clearing the draft, so a rejected send never eats what the
    // user typed. The thunk re-checks the same conditions, so a rapid second
    // Enter cannot open two turns either.
    const { valid, value } = validateMessage(input);
    if (!valid || sending || awaitingConsent) return;
    setInput("");
    await dispatch(sendChatMessage({ text: value, model }));
    refocus();
  }

  const handleKey = e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <ChatTemplate
      sidebarOpen={sidebarOpen}
      sidebar={
        <Sidebar
          user={user}
          sessions={sessions}
          activeSession={activeSession}
          model={model}
          onNewChat={newChat}
          onSelectSession={session => dispatch(openSession({ session }))}
          onDeleteSession={handleDeleteSession}
          onLogout={() => dispatch(logout())}
          error={sessionsError}
          onRetry={() => dispatch(fetchSessions())}
        />
      }
      header={
        <ChatHeader
          model={model}
          activeSession={activeSession}
          onToggleSidebar={() => setSidebarOpen(v => !v)}
        />
      }
      composer={
        <MessageInput
          value={input}
          onChange={setInput}
          onKeyDown={handleKey}
          onSend={send}
          sending={sending}
          awaitingConsent={awaitingConsent}
          inputRef={inputRef}
        />
      }
    >
      <MessageList
        messages={messages}
        sending={sending}
        model={model}
        awaitingConsent={awaitingConsent}
        onFeedback={(msg, rating) => dispatch(rateMessage({ msg, rating, userId: user.user_id }))}
        onConsentYes={async () => { await dispatch(acceptConsent({ model })); refocus(); }}
        onConsentNo={async () => { await dispatch(declineConsent({ model })); refocus(); }}
        onSuggestion={s => { setInput(s); refocus(); }}
        endRef={messagesEndRef}
        error={chatError}
        onDismissError={() => dispatch(chatErrorCleared())}
      />
    </ChatTemplate>
  );
}
