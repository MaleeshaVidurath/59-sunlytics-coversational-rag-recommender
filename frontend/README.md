# Sunlytics Frontend — Architecture

React SPA for the Sunlytics conversational fashion recommender. It talks to the M3
API (`localhost:8000`) for everything, and to the M2 image service (`localhost:8001`)
for product photos.

**Stack:** React 18 · Vite 5 · Redux Toolkit 2 · react-redux 9. No router, no CSS
framework, no component library.

```bash
npm install
npm run dev       # vite dev server
npm run build     # production build to dist/
npm run preview   # serve the built output
```

---

## 1. Directory layout

The UI follows **atomic design**; everything that is not UI sits beside it as a
flat sibling.

```
src/
├── main.jsx                  React root — mounts <Provider> and imports global.css
├── App.jsx                   22-line gate: login → model select → chat
│
├── components/
│   ├── atoms/                context-free primitives (8)
│   │   Button  IconButton  Badge  Avatar  Wordmark
│   │   ModelBadge  ProductImage  TypingIndicator
│   │
│   ├── molecules/            small groups of atoms (9)
│   │   MetaBadge  CorrectionNotes  ConsentButtons  FeedbackButtons
│   │   WhyList  ProductCard  SessionListItem  ModelOptionCard  MessageInput
│   │
│   ├── organisms/            meaningful page sections (6)
│   │   Message  MessageList  Sidebar  ChatHeader  CustomerPicker  ModelGrid
│   │
│   └── templates/            layout shells, no data (2)
│       CenteredTemplate  ChatTemplate
│
├── pages/                    state owners, wired to the store (3)
│   LoginPage  ModelSelectPage  ChatPage
│
├── store/
│   ├── index.js              configureStore + localStorage listener middleware
│   └── slices/               authSlice  modelSlice  sessionsSlice  chatSlice
│
├── services/                 all network access (5)
│   http.js  authService.js  sessionService.js  chatService.js  feedbackService.js
│
├── utils/                    pure helpers
│   time.js  labels.js  constants.js
│
└── styles/
    theme.js                  the `C` colour palette
    global.css                keyframes, scrollbar, select styling
```

### Dependency direction

Imports only ever point **downward**. An atom never imports a molecule.

```
pages  →  templates  →  organisms  →  molecules  →  atoms
  ↓                          ↓            ↓           ↓
store  ──────────────────────┴────────────┴───────────┘  (styles / utils only)
services
```

**The rule that keeps the lower tiers reusable: only organisms and pages may touch
the store.** Atoms and molecules receive everything through props. If a molecule
needs `useSelector`, it is really an organism.

In practice the codebase is currently stricter than that rule: `useSelector` /
`useDispatch` appear **only** in `App.jsx` and the three pages. Every organism is
props-driven too, which makes all 25 components renderable in isolation without a
`<Provider>`. Reaching for the store inside an organism is allowed when prop-drilling
gets genuinely painful — but it costs that property, so prefer passing props.

### Which tier does new code belong to?

| | Test | Examples here |
|---|---|---|
| **atom** | No app concepts. Reused ≥2 places. | `Button`, `Badge`, `Avatar` |
| **molecule** | A few atoms forming one small unit. | `ProductCard`, `MessageInput` |
| **organism** | A page section. May read the store. | `Sidebar`, `MessageList` |
| **template** | Layout slots only, never data. | `ChatTemplate` |
| **page** | Owns state, dispatches, composes. | `ChatPage` |

Atoms are extracted **only where the pattern already repeats**. One-off styling stays
inline in its parent — see the send button in `MessageInput`, which is deliberately
*not* an `IconButton` because it is a filled square with its own disabled logic,
unlike the three transparent glyph buttons `IconButton` covers.

---

## 2. State management

### What lives in Redux, and what does not

Redux holds state that **crosses components**. Everything else stays local.

| In the store | Why |
|---|---|
| `auth.user` | Read by the gate, the sidebar, and every thunk that needs `user_id` |
| `model.selected` | Gates routing, filters the sidebar, changes card headings |
| `sessions.items` / `activeId` | Sidebar writes it, chat reads it |
| `chat.messages` / `sending` / `awaitingConsent` | Written from thunks, read across the chat UI |

| Deliberately `useState` | Why |
|---|---|
| `input` (the message draft) | Changes every keystroke — in the store it would re-render the whole transcript per character |
| `sidebarOpen` | Pure view toggle, nothing else reads it |
| `imgFailed` / `preview` in `ProductCard` | Per-card view state |
| Login form fields | Nothing outside `LoginPage` reads a half-filled form |
| `messagesEndRef`, `inputRef` | DOM refs, not state |

### Store shape

```js
{
  auth:     { user: null | { user_id, customer_id, age, purchase_summary, ... } },
  model:    { selected: null | "m1" | "m2" | "m3" },
  sessions: { items: [...], activeId: null | "sess_xxx" },
  chat:     { messages: [...], sending: false, awaitingConsent: false },
}
```

### The slices

**`authSlice`** — `loggedIn(user)`, `loggedOut()`. Initial state is read from
`localStorage` so a refresh does not sign the user out.
Reducers stay pure: the `localStorage` write/remove is done by a
`createListenerMiddleware` listener in `store/index.js`, not inside the reducer.

**`modelSlice`** — `modelSelected(id)`, `modelReset()`. Deliberately **not**
persisted: one model per session, per login, so every fresh visit makes the choice
explicit. Listens to `loggedOut` and clears itself, so the next user to sign in on
this browser does not land in someone else's model.

**`sessionsSlice`** — `sessionsReceived`, `activeSessionSet`, `activeSessionCleared`
plus 3 thunks (below). A failed session-list read intentionally leaves the previous
list in place: a stale sidebar is less disruptive mid-conversation than an empty one.

**`chatSlice`** — `messagesSet`, `messageAppended`, `consentAwaited`,
`consentCleared`, `feedbackRecorded` plus 7 thunks. Also listens to
`activeSessionCleared` (empty the transcript) and `loggedOut` (reset everything).

### Cross-slice coordination

Slices react to each other's actions via `extraReducers` rather than one slice
importing another's reducer:

```
auth/loggedOut          → model, sessions and chat all reset themselves
sessions/activeSessionCleared → chat empties the transcript
```

`chatSlice` imports action creators *from* `sessionsSlice` (one direction only) —
never the reverse, which would create a circular import.

---

## 3. Async thunks

All 10 use `createAsyncThunk`. `authSlice` and `modelSlice` have none — they are
pure synchronous state.

| Thunk | Slice | What it does |
|---|---|---|
| `fetchSessions` | sessions | Loads the sidebar list |
| `refreshSessionsUntilPresent` | sessions | Retries until a new session appears (see below) |
| `removeSession` | sessions | Deletes, clears active id if it was open, reloads |
| `openSession` | chat | Loads a stored transcript, sets the active id |
| `refreshTranscript` | chat | Re-reads the transcript after a catalogue revision |
| `startNewChat` | chat | Clears the server pointer and the open transcript |
| `sendChatMessage` | chat | The main turn (see the sequence below) |
| `acceptConsent` | chat | Answers "yes" and shows the reply |
| `declineConsent` | chat | Answers "no" silently |
| `rateMessage` | chat | Optimistic thumbs up/down → RL collector |

### Only two thunks drive the typing indicator

`sending` is wired to `pending`/`fulfilled`/`rejected` for **`sendChatMessage` and
`acceptConsent` only**. `declineConsent` is excluded on purpose: it still calls the
backend so the server clears its pending-consent flag, but no reply is coming, and a
typing indicator for a reply that never arrives would be a lie. That is why Yes and
No are two thunks rather than one with a boolean flag.

### `sendChatMessage` — the critical path

```
1. append the user's turn                     (messageAppended)
2. read sessions.activeId from getState()
3. POST /api/chat
     force_new_session: !activeId    ← see below
     selected_model:    model
4. adopt the new session id          (activeSessionSet)  ← before appending the reply
5. append the assistant's turn       (messageAppended)
6. if the reply is a consent question → consentAwaited()
7. if res.revisions is non-empty     → await refreshTranscript(...)
8. dispatch refreshSessionsUntilPresent(...)   ← NOT awaited
```

Four details here are load-bearing:

- **`force_new_session: !activeId`** — no session open in the UI means the model was
  just picked, so this is the first message of a fresh chat. Without `force_new` the
  backend resumes whatever session its Redis pointer still holds, and *that* session's
  model lock overrides the model the user just selected.
- **Step 4 before step 5** — anything keyed off the open session must see the new id
  before the reply lands.
- **Step 7** — a catalogue value changed mid-conversation. The revision annotates
  messages sent *earlier*, and only the server knows which turns quoted the old value,
  so the whole transcript is re-read. Messages are never rewritten: they were accurate
  when sent, and silently editing history would be the dishonest fix. The correction
  appears as an amber note under the message it applies to.
- **Step 8 is not awaited** — the session is written to MongoDB as the turn completes,
  so an immediate read can miss it. `refreshSessionsUntilPresent` retries 3× at 600ms
  and then accepts whatever the server returned. Awaiting it would hold the composer
  disabled for up to 1.8s after the reply already arrived.

`sendChatMessage` also uses a `condition` option, so it will not fire when the text is
empty, a send is already in flight, or a consent question is outstanding — a fast
double-Enter cannot open two turns.

### Known rough edge

`sendChatMessage` catches its own errors and dispatches an error message, so it always
*fulfills* — `rejected` never actually fires for it and is wired only as a safety net.
The more idiomatic shape would use `rejectWithValue` and append the error turn from
`extraReducers`. The current form mirrors the original `try/catch/finally` exactly and
was kept to avoid re-specifying failure behaviour during the refactor.

---

## 4. Services layer

All network access goes through `services/`. No component calls `fetch` directly.

```
http.js           BASE = http://localhost:8000
                  M2_IMAGE_BASE = http://localhost:8001
                  request() / requestJson()
```

| Endpoint | Method | Service |
|---|---|---|
| `/api/auth/customers` | GET | `authService.getCustomers` |
| `/api/auth/login` | POST | `authService.login` |
| `/api/sessions?user_id=` | GET | `sessionService.getSessions` |
| `/api/sessions/{id}?user_id=` | GET | `sessionService.getSessionHistory` |
| `/api/sessions/{id}?user_id=` | DELETE | `sessionService.deleteSession` |
| `/api/sessions/new?user_id=` | POST | `sessionService.startNewSession` |
| `/api/chat` | POST | `chatService.sendMessage` |
| `/api/rl/feedback` | POST | `feedbackService.submitFeedback` |
| `/api/images/{article_id}` | GET | direct `<img src>` on port **8001** |

**Error handling is intentionally per-endpoint, not uniform.** `request()` takes an
opt-in `errorMessage`: endpoints that have always thrown on non-2xx (login, chat,
delete) still throw; endpoints that never checked `res.ok` (customers, session list,
history) still parse the body regardless; `submitFeedback` swallows failures entirely,
because the rating is already reflected in the UI and a lost training signal must not
surface as a chat error. Standardising this would be a real improvement — it is
deliberately left as a separate decision rather than changed silently.

---

## 5. Styling

Inline style objects, with the palette centralised in `styles/theme.js` and exported
as `C`. Global rules (the `bounce` keyframes, scrollbar, `select` options) live in
`styles/global.css`, imported once from `main.jsx`.

Playfair Display is loaded by a `<link>` in `index.html`.

CSS modules were considered and deliberately deferred: converting ~1000 lines of
inline styles is a restyle with real regression risk and no structural benefit.

---

## 6. Routing

There is none, on purpose. The flow is a linear auth gate, not URL-driven:

```
App.jsx:  !user → <LoginPage>    !model → <ModelSelectPage>    else → <ChatPage>
```

Adding `react-router` would change refresh and back-button behaviour. It is the right
move only if chats need to be deep-linkable.

---

## 7. Gotchas

**Chat state outlives `ChatPage`.** Before Redux, clicking "New Chat" unmounted
`ChatPage` and its local state reset for free. Store state does not. `startNewChat`
therefore clears the transcript explicitly, and `activeSessionCleared` empties it
whenever the open session goes away. If you add another path back to model selection,
it must clear chat state too, or the next chat will open showing the previous
conversation.

**The consent flow depends on an exact string.** `CONSENT_TRIGGER` in
`utils/constants.js` must stay byte-identical to what the backend appends. The UI
matches on it to swap the composer for Yes/No buttons.

**Sessions are locked to their model.** The sidebar filters by
`selected_model`, because a session opened under M2 cannot be resumed under M3.

**Message ids from history are not always stable.** `mapHistoryMessages` falls back to
`Math.random()` when a stored turn has no `turn_id`, which makes a poor React key.
Worth fixing if history rendering ever misbehaves.

---

## 8. Verification

There is **no test runner configured**. Changes are currently verified by
`npm run build` plus a manual pass:

> login → select model → send a message → thumbs up → New Chat → reopen a session → delete a session → sign out

If you add a runner, Vitest fits the Vite setup with no extra config. The store and
the thunks are the highest-value things to cover — they hold the session/consent logic
described above and are testable without a browser by dispatching against a mocked
`fetch`.

---

## 9. React concepts used

Everything below is actually present in this codebase — no aspirational entries.

### Core

| Concept | Where |
|---|---|
| **Function components** | All 25 components + 3 pages. No class components anywhere. |
| **JSX** | Throughout. `.jsx` for anything rendering, `.js` for store/services/utils. |
| **Props** | The main data channel. All atoms/molecules/organisms are props-driven. |
| **Composition via `children`** | `CenteredTemplate`, `ChatTemplate`, `Avatar`, `Button`, `IconButton`, `Badge` |
| **Composition via slot props** | `ChatTemplate` takes `sidebar`, `header`, `composer` as JSX *props* rather than children — it needs four independent slots, and `children` only gives you one. See `ChatPage.jsx:79-108`. |
| **Lists and keys** | 9 `.map()` renders — messages, sessions, product cards, model options, why-reasons |
| **Conditional rendering** | `&&` and ternaries throughout; `App.jsx` gates the whole app with three early returns |
| **Early return `null`** | `FeedbackButtons` (no `recommendation_id`), `CorrectionNotes` (no corrections), `WhyList` (no reasons) — render nothing rather than an empty wrapper |
| **Fragments (`<>…</>`)** | `Sidebar`, `CustomerPicker`, `MessageList` — group siblings without a DOM node |
| **Controlled components** | `<textarea>` in `MessageInput`, `<select>` in `CustomerPicker` — value comes from state, changes flow back through `onChange` |
| **Event handling** | `onClick`, `onChange`, `onKeyDown`, `onInput`, `onMouseEnter/Leave`, `onError` |
| **Lifting state up** | The message draft lives in `ChatPage`, not `MessageInput`, because `send()` needs it |
| **Derived state** | Computed during render, never stored: `canSend` in `MessageInput`, the model-filtered session list in `Sidebar`, `showConsent` in `Message` |
| **Props spreading + rest** | `Button`, `IconButton`, `Badge` take `...rest` so callers can pass `title`, `disabled`, etc. |
| **Default prop values** | Default parameters (`size = 32`, `style = {}`) — the legacy `defaultProps` API is not used |

### Hooks

| Hook | Count | Used for |
|---|---|---|
| `useState` | 21 | Local view state only — draft text, sidebar toggle, hover, image-failed, login form |
| `useEffect` | 5 | Fetch on mount (`fetchSessions`, `getCustomers`), scroll-to-bottom on new messages |
| `useRef` | 3 | DOM handles — `messagesEndRef` for scrolling, `inputRef` for refocusing |
| `useSelector` | 11 | Reading store state (pages only) |
| `useDispatch` | 6 | Dispatching actions and thunks (pages only) |

**Effect dependency arrays matter here.** The scroll effect depends on
`[messages, sending]` so it fires on every new turn *and* when the typing indicator
appears. The session-load effect depends on `[dispatch, user.user_id]` so it runs
once per user rather than on every render.

### State management

- **Redux Toolkit**: `configureStore`, `createSlice`, `createAsyncThunk`,
  `createListenerMiddleware`
- **Immer** (built into RTK) — reducers *look* mutable (`state.messages.push(...)`)
  but produce immutable updates. Only valid inside `createSlice`; anywhere else,
  mutating state is still a bug.
- **Context** — used indirectly. `<Provider>` is a context provider; `useSelector`
  reads through it. No hand-written `createContext` in this codebase.
- **Optimistic UI** — `rateMessage` writes the thumbs rating into the message
  *before* the network call, so the button responds instantly.

### Deliberately NOT used

Listed so their absence reads as a decision rather than an oversight.

| Not used | Why |
|---|---|
| `React.memo`, `useMemo`, `useCallback` | No measured performance problem. The transcript is short and re-renders are cheap. Adding memoisation before profiling adds complexity and stale-closure bugs for nothing. |
| `forwardRef` | `MessageInput` receives `inputRef` as a plain prop instead. Works fine and is simpler; switch to `forwardRef` only if these become a shared library. |
| `useReducer` | Redux already owns the complex state; local state is simple enough for `useState`. |
| `useContext` (hand-rolled) | Redux covers cross-component state. A second mechanism would fragment it. |
| Class components | Nothing needs them. |
| Error boundaries | **A genuine gap, not a preference.** A render error currently blanks the page. Worth adding one around `ChatPage`. Requires a class component — it is the one thing hooks still cannot do. |
| `Suspense` / `lazy` | The bundle is ~200 kB; code-splitting is not yet worth it. |
| `createPortal` | The image lightbox uses `position: fixed` inside the tree, which is sufficient here. |
| `StrictMode` | Not enabled. Turning it on would double-invoke effects in dev and surface any accidental side-effects in render — worth doing, but it may reveal existing issues, so it is a deliberate follow-up rather than a silent change. |

---

## 10. Error handling & validation

### Principles

1. **Every failure becomes an `ApiError`.** Callers never distinguish a `TypeError`
   from `fetch` from a 500 from a JSON parse failure by string-matching.
2. **Every response status is checked.** No endpoint parses the body of a failed
   response.
3. **Every request has a deadline.** No `fetch` can hang forever.
4. **Errors reach the user.** `console.error` alone is not error handling.
5. **Validation lives in one place** — `utils/validation.js`, as pure functions.

### `ApiError` (`services/ApiError.js`)

```js
new ApiError(kind, message, { status, url, body, cause })
```

| `kind` | Cause |
|---|---|
| `NETWORK` | Server unreachable / CORS — `fetch` rejected |
| `TIMEOUT` | Aborted by our own deadline |
| `HTTP` | Server replied with a non-2xx |
| `PARSE` | 2xx with a body we could not read |

- **`.isRetryable`** — true for network, timeout, 429 and 5xx. Deliberately false
  for other 4xx: a rejected request stays rejected.
- **`.userMessage`** — display text that never leaks a stack, URL or raw payload.
- **`.toPayload()`** — plain `{ kind, status, message }` for Redux, because `Error`
  instances are not serialisable.

### HTTP layer (`services/http.js`)

- `AbortController` deadline on every request, default **30s** (a chat turn runs an
  LLM plus retrieval).
- Bodies read with `res.text()` then parsed defensively — error responses are often
  HTML, not JSON.
- **Opt-in retries with exponential backoff** (300ms → 600ms → 1200ms), only for
  retryable failures. Applied to idempotent GETs (`retries: 2`).
- **`POST /api/chat` is never retried** — it is not idempotent, and retrying after a
  timeout would append a turn the user never sent. Same for `DELETE`.
- All path parameters go through `encodeURIComponent`.

### Where errors surface

| Failure | What the user sees |
|---|---|
| Chat turn fails | Assistant turn with the real cause ("Cannot reach the server…"), `isError: true` |
| Session list fails | `ErrorBanner` in the sidebar with a **Retry** button; the previously loaded list stays |
| Opening a chat fails | Dismissible `ErrorBanner` above the transcript |
| Transcript refresh fails | Same banner — non-fatal, the reply itself arrived |
| Render throws | `ErrorBoundary` fallback with *Try again* / *Reload* (stack shown in dev only) |
| Feedback POST fails | **Nothing** — the one deliberate exception. The rating is already shown, and a lost RL signal must not become a chat error. |

`ErrorBoundary` is the only class component in the codebase — there is no hook
equivalent of `componentDidCatch`. It wraps `<Provider>` in `main.jsx` so it also
catches a store-level failure. It does **not** catch event handlers or async code;
those are covered by the `ApiError` path above.

### Validation (`utils/validation.js`)

| Rule | Behaviour |
|---|---|
| Empty / whitespace-only message | Send disabled. No error text — an empty box is not a mistake worth announcing. |
| Message > `MAX_MESSAGE_LENGTH` (2000) | Inline error, red border, send disabled |
| Approaching the limit (last 200 chars) | Live `1850/2000` counter appears |
| Customer profile not chosen | Sign-in disabled |

The textarea deliberately **accepts** overlong text rather than truncating at the
keystroke — silently eating part of a paste is worse than explaining the limit. The
same `validateMessage` runs in `ChatPage.send()` before the draft is cleared, so a
rejected send never loses what the user typed.

### Behaviour change from the original

Before this, six endpoints never checked `res.ok`. A 500 from `/api/sessions` was
parsed as `data.sessions || []` and became **an empty sidebar with no error** — the
user could not tell "no chats yet" from "the backend is down". Those failures are now
visible. This is the intended fix, but it does mean errors appear where the UI
previously showed nothing.

---

## 11. Authentication

Sign-in uses **JWT access + rotating opaque refresh tokens delivered as
`httpOnly` cookies**. The full model, threat coverage and known gaps are in
[`../SECURITY.md`](../SECURITY.md); this section covers only what the frontend does.

### There is no token in JavaScript

The session is **not** in `localStorage`, Redux, or any variable. Access and
refresh tokens are `httpOnly` cookies the page cannot read, so an injected
script has nothing to steal. The one readable cookie is the CSRF token, which is
readable by design — the double-submit pattern needs the page to echo it back in
an `X-CSRF-Token` header, something a cross-origin attacker cannot do.

### Bootstrap

Because the page cannot read its own session, `App.jsx` does not know on load
whether anyone is signed in. It dispatches `bootstrapSession()`
(`GET /api/auth/me`) and holds a neutral splash until the answer arrives —
rendering the login screen first would flash it at users who *are* signed in.

```
status:  "bootstrapping"  ->  "anonymous" | "authenticated"
```

### The 401 interceptor

Access tokens last 15 minutes. `services/http.js` turns that into a non-event:

```
request -> 401 -> POST /api/auth/refresh -> replay original request -> 200
```

**Concurrent 401s share one refresh.** This is not an optimisation. Refresh
tokens rotate and reuse is treated as theft, so two parallel refreshes would
land the second on an already-rotated token, revoke the whole family, and log
the user out *because the page was busy*. The single-flight promise in
`refreshSession()` prevents that, and it is cleared **synchronously** — a
deferred clear leaves an already-resolved `false` in place, and the next 401
then "fails" a refresh it never attempted.

If refresh itself fails, `setSessionExpiredHandler` tells the store, which
clears the user, transcript, sidebar and model choice and returns to sign-in.

### Requests carry no identity

`user_id` and `customer_id` are absent from every payload and query string — the
server derives them from the token. Passing a client-supplied `user_id` used to
let anyone read or delete another user's chats.

### Screens

| | |
|---|---|
| `LoginPage` | Username + password. Credentials for the 250 research personas are in the gitignored `credentials/seeded_accounts.csv`. |
| `RegisterPage` | Creates a **cold-start** account: no linked H&M persona, so no purchase history. The form says so, because early recommendations genuinely differ. |
| `CredentialsForm` | Shared by both. Handles confirm-match and the 12-character minimum client-side; the server enforces the real policy. |

`CustomerPicker` was deleted — picking a customer from a list is no longer how
you sign in.

### Testing

```bash
npm run test:auth
```

Drives the real store and services against a mock that emulates the server's
cookie and 401 semantics: httpOnly invisibility, CSRF echo, expiry, refresh
rotation, and session death.
