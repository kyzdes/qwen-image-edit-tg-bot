# Building Great Bots: A Framework for Wrapping a Model/API into a Polished Telegram Bot

## What this playbook is for

This is a repeatable method for turning a model or API — a Hugging Face Space, a raw model, a REST service — into a polished, production Telegram bot that runs 24/7. It was extracted from one real build: wrapping the `productowner/Qwen-Image-Edit-2509-LoRAs-Fast-v2` Gradio Space (text-prompt image editing with selectable LoRAs) into an allowlisted, Dokploy-hosted, long-polling bot built on `python-telegram-bot` v21. Every fact below was learned by doing it.

**The one-line philosophy:** *Verify empirically, never guess — observe the contract, design the flow on paper, respect the platform's real constraints, and prove the thing works before you call it done.*

## Pipeline at a glance

```
Discover ──> Design ──> Menus ──> Data ──> Backend ──> Build ──> Deploy
```

Each phase consumes the artifact the previous one produced — the chain only works end to end:

1. **Discover** — Read-only reconnaissance: classify the target, pin its exact I/O contract, map capabilities and the cost/quota model — all empirically. → produces a **contract sheet** (exact endpoint, param shapes, feature catalog, quota math).
2. **Design** — Capture intent on paper: explore context, answer the architecture-defining forks, run the safety gate, write the UX flow. *Consumes the contract sheet* (the input shape and quota model constrain the UX). → produces a **one-page approved flow** (fork answers + safety outcome + flow diagram).
3. **Menus** — Turn the feature surface into inline keyboards, a compact callback scheme, and a free-text state machine; nail the Telegram interaction gotchas. *Consumes the flow diagram* (it pre-decided the menus and where free text routes).
4. **Data** — Model per-user session/settings, adapt input to the backend's exact shape, respect output limits, self-track usage, persist durably. *Consumes the contract sheet's input shape and quota facts, plus the awaiting-states named in Menus.*
5. **Backend** — Wrap the model in a resilient client: version-safe auth, off-loop calls, reconnect-once, an error taxonomy, honest quota wording. *Consumes the contract sheet's endpoint signature and quota model.*
6. **Build** — Contract → parallel modules → adversarial review → orchestrator-run verification, with evidence before "done." *The shared contract here is the union of every prior phase's decisions.*
7. **Deploy** — Package as one Docker container, ship to a VPS via Dokploy long-polling, lock it down, persist state, and run the commit→deploy→monitor→verify loop. *Consumes the persistence and audience decisions from Design and Data.*

---


## Phase 1 — Discover & Analyze What You're Wrapping

Before you write a single line of bot code, you must know exactly what you are calling: *what kind of thing it is*, its *exact I/O contract*, the *real knobs it exposes vs. what is hardcoded*, and its *hardware/cost/quota model*. Every fact in this phase must be **discovered empirically, not assumed** — guesses here become bugs that only surface in production. The one time we trusted a UI label over the source (the GPU-size question), we were wrong.

This phase is pure reconnaissance: read-only, no commitments. **Its output is a written "contract sheet"** — the exact endpoint and ordered params, each param's *shape*, the verbatim feature catalog, what's fixed, and the quota math — that the Design (Phase 2), Data (Phase 4), and Backend (Phase 5) phases all build against.

### 1.1 Classify the target: model vs. app/Space vs. REST

The single most important early question, because it determines *everything* about how you call it. These are not interchangeable:

| Target type | What it really is | How you call it | What you must discover |
|---|---|---|---|
| **Raw model** (weights) | safetensors + a model card | a serving stack you provide, or a hosted inference endpoint | the inference code, tokenizer/preprocessing, VRAM fit |
| **App / Gradio Space** | a *running application* wrapping a model, with its own preprocessing, defaults, and UI logic | `gradio_client` against its named endpoints | the function signature behind each endpoint — **not** the model's native API |
| **REST / hosted API** | an HTTP service with documented routes | HTTP client | auth scheme, request/response schemas, rate limits |

Our target was a **Gradio Space** (`productowner/Qwen-Image-Edit-2509-LoRAs-Fast-v2`), *not* the raw Qwen-Image-Edit model. This distinction is load-bearing: the Space had already baked in a fixed negative prompt, auto-resolution (~1024px long side), a fixed LoRA weight of 1.0, and a specific model stack. Had we treated it as the raw model, we'd have rebuilt all of that — and gotten different results than the Space produces.

> **Rule:** find out whether you're calling the *app* or the *model*. An app has opinions (preprocessing, defaults, fixed params) you must inherit, not reinvent.

For a Space, inspect the app shell first:

```bash
hf spaces info <owner>/<space> --format json   # hardware, sdk, file list
hf download <owner>/<space> app.py requirements.txt --repo-type space
```

Then **actually read `app.py` and `requirements.txt`.** This is where the truth lives — see 1.3.

### 1.2 Pin the exact I/O contract — empirically

Do not infer the contract from the UI or the README. Derive it from the wire.

For a Gradio Space, `view_api()` is the authoritative source:

```python
from gradio_client import Client
client = Client("<owner>/<space>")
client.view_api()   # prints every endpoint + ordered params + return types
```

For us this revealed the real endpoint:

```
/infer(image_b64, prompt, lora_adapter, seed, randomize_seed, guidance_scale, steps)
       -> (image_filepath, used_seed)
```

The **critical, non-obvious detail**: `image_b64` is a **base64 data-URI string**, not a file upload. You will never learn that from a screenshot of the UI — and getting it wrong means every call fails. Confirm input *shape* and output *shape* with a real call, then inspect the actual artifact:

```python
out = client.predict(image_b64=data_uri, prompt="...", lora_adapter="...",
                     seed=0, randomize_seed=True, guidance_scale=1.0, steps=4,
                     api_name="/infer")
# out -> (local_filepath, used_seed); OPEN the returned image and look at it.
```

We did exactly this and visually inspected the returned image to confirm the Space did what we expected. **A 200 response is not proof of correct behavior — the artifact is.**

For a REST API the analogue is: read the OpenAPI/Swagger spec if it exists, then send one real request and capture the full response (body *and* headers — headers matter, see 1.4).

### 1.3 Map real capabilities: knobs vs. fixed

Reading the source gives you the *feature surface* — and, just as importantly, the boundary between what the user can tune and what is locked. Both halves go into your bot's UX. From `app.py` we extracted:

**Tunable (expose these as bot controls):**
- **11 LoRA adapters** — the exact adapter strings, each with a sensible default prompt and example direction variants (camera angles, light directions). These became the bot's feature menu.
- **Generation params + defaults:** `steps` (default 4), `true_cfg`/`guidance` (default 1.0), `seed`, `randomize_seed`.

**Fixed (do NOT expose; inherit silently):**
- negative prompt — hardcoded in the Space
- resolution — auto, ~1024px long side
- LoRA weight — fixed at 1.0

**Context you must know but can't change:**
- model stack: base Qwen-Image-Edit-2509 + a Rapid-AIO fp8 transformer, bf16, Flash-Attention-3
- `@spaces.GPU(size="xlarge")` decorator (this is the cost model — see 1.4)
- launched with `mcp_server=True`

> **Why this matters:** capturing the exact adapter strings and defaults from source meant the bot reproduced the Space's intended results on the first try, and the menu offered *only* knobs that actually do something. Exposing a "resolution" slider that the Space ignores would be a lie to the user. **These verbatim strings flow straight into the shared contract (Phase 6.1) — copy them once, never retype them.**

**Check whether the target model is gated — a gate is a hard wall, not a warning.** If your target wraps (or is) a model that is *gated* on the Hub, you cannot call it, prefetch it, or download its weights until a human has accepted the license **in the web UI** — there is no programmatic accept. The consent form (`Agree and access` on the model page) is a UI-only action; an API call against a gated repo just returns `403 Cannot access gated repo`. We hit this with `kpsss34/FHDR_Uncensored` and `black-forest-labs/FLUX.1-dev` (both `gated=auto`): every prefetch 403'd until the account clicked agree on the page. So during Discovery, check it explicitly:

```python
from huggingface_hub import model_info
model_info("<owner>/<repo>").gated   # False, or "auto"/"manual" if gated
```

If it's gated, flag that a human must open the model page and accept *before* anything will work — and make sure the eventual error path (in the bot or the backend Space) names the **exact model id and its page** to accept, so the fix is obvious instead of a bare 403.

### 1.4 Map the hardware / cost / quota / rate-limit model

This is where naive wrapping silently fails or runs up a bill. Build a precise mental model, and **prove the parts you can't see.**

**Where the cost knobs actually live.** On HF ZeroGPU (shared NVIDIA RTX Pro 6000 Blackwell here), the GPU size is set **only in the `@spaces.GPU(size=...)` decorator in code** — `large` = 48GB / 1× quota multiplier, `xlarge` = 96GB / 2× quota multiplier. It is **not** a per-request parameter and **not** in the hardware dropdown. We initially assumed the UI dropdown controlled it; reading the source corrected us. *Find the real knob, don't trust the obvious one.*

**The quota math (per account, not per call):**

| Tier | Daily GPU budget | Reset |
|---|---|---|
| PRO | 40 min | 24h after first use |
| Free | 5 min | 24h after first use |
| Anonymous | 2 min | 24h after first use |

Overage is billed from credits at ~$1 / 10 min. Quota is consumed as `actual_seconds × size_multiplier` (xlarge = ×2), so the same wall-clock generation burns twice the budget at `xlarge` as at `large`.

**Prove what the provider does NOT expose.** We captured *all* HTTP response headers during a live generation to confirm there is **no public API for remaining quota** — only generic rate-limit headers. The exact remaining-seconds and reset time appear **only in the quota-exceeded error text**. This empirical proof drives two downstream decisions: the bot must self-track usage (Phase 4 §4) and must parse the error text for the authoritative numbers (Phase 4 §4b / Phase 5 §5.6), because no clean API exists. (Don't take "there's no API" on faith — capture the headers and *show* there isn't.)

**Judge local-run feasibility deterministically.** Instead of guessing whether it would OOM, we summed the weight files against VRAM:

```
transformer (fp8)   20 GB
text-encoder        16.6 GB
VAE                 0.25 GB
-------------------------
total              ~37 GB   vs. available GPU VRAM  -> deterministic verdict
```

A subtraction beats a guess.

### 1.5 Authenticate first — and never print the secret

Before any of the above, pull provider credentials from a secrets manager and authenticate the CLI **without ever echoing the value.** Auth is not a formality: it directly changes the quota you get (anonymous ≈ 2 min vs. PRO 40 min for the same Space). Discovery done anonymously will mislead you about real-world limits. Confirm which account/tier you authenticated as before trusting any quota number you observe.

**If the backend is private, the token is mandatory just to *reach* it — not merely to raise your quota.** The quota point above assumes a *public* Space, where anonymous still connects (just with a tiny budget). A **private** Space is different in kind: `gradio_client` cannot connect to it *at all* without `token=` — you don't get a small quota, you get no connection. So when your target is private, discover that fact now and treat the auth token as a hard requirement of the bot's config, not an optional tier upgrade. (Carry this forward to Backend/Deploy: a missing token there is a connection failure, not a degraded one.)

**If you're also building the backend Space yourself (not just wrapping someone else's), the contract is yours to define — but ZeroGPU and Gradio have sharp edges to discover up front.** Two we hit: (1) on **ZeroGPU**, each `@spaces.GPU` call runs in a *forked worker*, so a module-level pipeline cache does **not** persist reliably across calls — a multi-model Space must load the selected model *fresh per call* (prefetch all model files at module scope so the GPU window only reads from local disk). (2) **Gradio component kwargs are version-specific** — passing a newer kwarg the Space's Gradio doesn't support (e.g. `gr.Gallery(show_download_button=True)`) crashes the Space at startup with `RUNTIME_ERROR`. Note these now so they shape the backend you design; the full write-ups are the canonical Gotchas catalog entries (#19 ZeroGPU load-fresh-per-call, #20 version-specific Gradio kwargs).

### Discovery checklist

Work top to bottom. Each box is a fact you *verified*, not assumed.

```
AUTH & CREDS
[ ] Creds pulled from secrets manager; CLI authenticated; secret never printed
[ ] Confirmed which account/tier you're authing as (it changes quota)
[ ] Is the backend PRIVATE? If so, token is MANDATORY just to connect — flag it as required config

CLASSIFY
[ ] Determined target type: raw model | app/Space | REST API
[ ] If a Space: pulled `hf spaces info --format json` (hw, sdk, files)
[ ] Downloaded and READ app.py + requirements.txt (or the model card / OpenAPI spec)
[ ] Checked model_info(repo).gated — if gated, a HUMAN must accept the license in the web UI first

I/O CONTRACT (empirical)
[ ] Ran view_api() (Space) / read OpenAPI (REST) — got the exact endpoint + ordered params
[ ] Noted exact input SHAPE of each param (e.g. base64 data-URI STRING, not file upload)
[ ] Noted exact output shape (e.g. (filepath, used_seed))
[ ] Ran ONE real call and visually inspected the returned artifact — behavior confirmed

CAPABILITIES (from source)
[ ] Listed every tunable param + its DEFAULT (steps, cfg/guidance, seed, randomize…)
[ ] Listed the full feature catalog with EXACT strings (e.g. 11 LoRA adapter ids + default prompts + variants)
[ ] Listed what is FIXED/hardcoded (negative prompt, resolution, LoRA weight) — do not expose these
[ ] Recorded the model stack & any decorators (e.g. @spaces.GPU(size=...))

HARDWARE / COST / QUOTA
[ ] Found WHERE the cost knob lives (decorator? param? dropdown?) — verified, not assumed
[ ] Wrote down the quota model: budget per tier, unit (GPU-sec × multiplier), reset window, overage price
[ ] Captured ALL response headers on a live call — PROVED whether a remaining-quota API exists
[ ] Found where remaining/reset numbers DO appear (here: only in the quota-exceeded error text)
[ ] If considering local run: summed weight-file sizes vs. VRAM — deterministic fit check
```

**The ethos of this phase:** verify empirically, don't guess. View the API; read the source; capture the headers; run one live call; *look at the output*. Everything you nail down here becomes the contract sheet that the flow design (Phase 2) and the backend wrapper (Phase 5) build against — and every fact you assumed instead of verified is a production bug waiting to happen.

---

## Phase 2 — Capture Intent and Design the Flow BEFORE Coding

The biggest risk in a model-to-bot project isn't a bad line of code — it's building the wrong shape. By the time you've wired keyboards, state, and a backend client to a flow that turns out to be wrong, you're rewriting the architecture, not patching it. Phase 1 told you what the backend *can* do; you have its contract sheet in hand. Phase 2 decides what the bot *will* do, and locks it down on paper before a single handler exists.

The discipline: **clarify first, design explicitly, gate for harm, then write down the flow as a contract.** In the build this happened entirely before code — explore the context, ask the architecture-defining questions, get the design approved, refuse the one harmful framing, and only then hand a written UX flow to the build phase.

### Step 1 — Explore the context, *then* ask

Run a real brainstorming/clarification step before proposing anything. But brainstorming blind wastes the user's time asking questions you could answer yourself. Inventory the context *first*, so every question you ask is one you genuinely can't resolve.

| Context signal | How you check it | Why it changes the design |
| --- | --- | --- |
| Is the workspace empty or existing? | List the working dir | Empty dir = greenfield; existing repo may pin language/framework |
| What credentials already exist? | Check the secrets manager (e.g. keys-keeper) | A reusable bot token or provider key may already be sitting there |
| Is there hosting already? | Check for an existing VPS / Dokploy instance | Decides whether "24/7 hosting" is a fresh provision or a slot you fill |
| What did Phase 1 reveal about the backend? | Re-read your contract-sheet notes | The input type and quota model constrain the whole UX |

Only after this do you ask the questions you couldn't answer from context.

### Step 1.5 — Decide the flow SHAPE: input-bearing vs prompt-only

Before the forks, settle one binary that the whole skeleton hangs on: does the bot **carry a user-supplied artifact through the flow, or not?** This shape is upstream of everything — the menu, the state machine, and the handlers all fall out of it differently. Pick it now, on paper, so Phase 3 (Menus) and Phase 4 (State) aren't building against an unstated assumption.

The two shapes seen across real builds:

- **Input-bearing** — the user supplies an artifact, you **store it**, then operate on it. Bot #1 was this: upload a photo → stash it as session state → every feature acts on that stored image. The flow needs a photo (or file/voice) handler that captures and persists the artifact *before* any generation, and free text later is contextual (a custom prompt or a typed setting), which is exactly what forces the awaiting-state machine.
- **Prompt-only** — there is **no upload**; the user picks a model and sends a text prompt, and the text *is* the input. Bot #2 (text-to-image) was this: pick a model → type a prompt → generate. There is no photo handler and nothing to stash; free text routes **straight to generation** as the prompt. The state machine is lighter because there's no stored artifact to thread through.

Two skeletons, side by side:

```
INPUT-BEARING                         PROMPT-ONLY
send <artifact>  → store it           pick a model
   ↓                                     ↓
feature menu (acts on stored input)   send text prompt  → that text IS the input
   ↓                                     ↓
pick feature / type custom prompt     generate
   ↓                                     ↓
generate (against stored artifact)    result
```

Don't blur the two: a prompt-only bot that grows a stray photo handler, or an input-bearing bot that forgets to persist the artifact, is the "wrong shape" failure this phase exists to prevent. Name the shape here; the menu + state machine follow from it. (This decision then *feeds* Fork 1 — it fixes whether "core user input" is a stored artifact or the prompt text itself.)

### The 4 architecture-defining forks (generalized)

These are the questions whose answers reshape the architecture — not cosmetic preferences. Ask *these*, get them approved, and most downstream decisions fall out. Ordered by impact.

**Fork 1 — How is the core user input handled, and does it need an extra model?** *(the biggest driver)*
This is the question that bends the entire architecture. In the build the input was an image the user sends, base64-encoded to match the Space's string contract — no extra model needed. But the decisive sub-question is whether you need *another* model in front of the primary one: e.g. a vision model to auto-understand the user's image, a transcription model for voice notes, an OCR pass on a document. An extra model means an extra API contract, extra latency, extra failure modes, and extra cost — so settle it first because everything (state shape, error taxonomy, quota math) depends on it.
- What is the primary input modality (photo / text / voice / document / file)?
- What exact shape does the backend want it in? (string data-URI vs file upload vs URL — you proved this empirically in Phase 1; don't re-guess here)
- Does interpreting that input require a *second* model before the primary call? If yes, that model is now part of the architecture, not an afterthought.

**Fork 2 — Credential source.**
- Is this a brand-new bot (new token from BotFather) or an existing one?
- Where does every secret live? In the build, a new token went straight into the secrets manager — never pasted into code or chat. Decide the storage *before* you need the value, so it never leaks into a transcript.

**Fork 3 — Hosting target.**
- Local long-poll for fast iteration, or VPS/Dokploy for 24/7? These aren't mutually exclusive — local for the dev loop, VPS for production was the actual path.
- This fork dictates packaging (Dockerfile, env handling), the persistence story (a mounted volume only matters if you're deploying), and the deploy loop you'll live in during Phase 7.

**Fork 4 — Audience: private allowlist or open?**
- Private (allowlist by user_id) or open to anyone? The build was allowlisted to specific users.
- This is not just an auth checkbox — it drives **quota and cost exposure**. An open bot on a metered backend (the GPU quota was per-account GPU-seconds/day) can drain your quota or rack up overage; a private allowlist caps who can spend it. Decide audience and cost-exposure together.

Checklist — don't start coding until each is a concrete, written answer:

- [ ] Primary input modality and its **exact backend shape** are named
- [ ] Decided whether a **second model** sits in front of the primary call
- [ ] Credential source named; storage location chosen; nothing will be echoed
- [ ] Hosting target chosen (and whether local-dev + prod differ)
- [ ] Audience chosen (allowlist vs open) **with** its quota/cost implication stated
- [ ] The design has been **explicitly approved** before any handler is written

### Step 2 — The safety / ethics gate

Run this gate at intent capture, before you design or build — it can kill a feature, and you don't want to discover that after wiring it up.

In the build, the gate fired: a request to turn the photo editor into a nudify / NCII tool was declined — and stayed declined when reframed as "educational." The move that works:

1. **Identify the harmful capability**, not just the harmful wording. Re-framing ("it's for education") doesn't change what the feature *does*.
2. **Decline the harmful use.**
3. **Redirect to the legitimate capabilities** the same backend genuinely offers, so the user still leaves with a buildable bot.

Gate checklist:

- [ ] Could this feature produce non-consensual, deceptive, or abusive output? If yes → decline.
- [ ] Did a re-frame ("educational", "just testing") change the *actual capability*? If no → the original decline still holds.
- [ ] Offered a concrete legitimate redirect using the same backend.

### Step 3 — Write down the explicit UX flow

Before building, draw the whole interaction as one linear flow. **This diagram is the deliverable that Phase 3 (Menus) and Phase 4 (State) implement against** — it pre-decides the menus, the state you'll need, and where free-text routing happens, so the parallel build doesn't drift. The build's flow, as a reusable template:

```
send <primary input>                      # e.g. photo
   ↓
feature menu                              # grid of capabilities from Phase 1
   ↓
pick a feature
   ├─ sets a sensible default prompt/params
   ├─ some features open a SUBMENU of directions (variants)
   └─ OR user types a custom prompt instead
   ↓
(optional) adjust settings                # steps / cfg / seed — within clamped limits
   ↓
Generate
   ↓
result sent with:
   ├─ an informative caption
   └─ action buttons: Repeat · Settings · New input · Menu
```

What makes this flow worth writing down (not just "the user sends a thing and gets a thing"):
- **Defaults are part of the flow.** Picking a feature *sets* a default prompt and params, so a user can generate immediately without typing — the custom-prompt path is an override, not the only path.
- **Submenus are conditional.** Only features with real variants (the build had direction variants like camera angles / light directions) open a submenu; flat features go straight to a default. Decide per-feature now so the keyboard layout is known before Phase 3.
- **The result is a launchpad, not a dead end.** Every result ships with action buttons (Repeat / Settings / New input / Menu) so the next iteration is one tap away — this directly shapes the result-action keyboards and the "again" callback you'll build later.
- **Name the free-text branches.** "User may type a custom prompt instead" and "set a seed by typing" mean free text is contextual — which is exactly what forces the awaiting-state machine you'll design in Phase 3 (modeled in Phase 4). Surfacing it here means those phases aren't a surprise.

Deliverable of Phase 2: a one-page, approved artifact stating the 4 fork answers, the safety-gate outcome, and this flow diagram. That artifact is the input to the shared contract in the build phase — design approved on paper is cheaper to change than design discovered in code.

---

## Phase 3 — Design Convenient Menus & Interactions (Telegram Inline UX)

A model behind a bot is only as good as the surface a user touches. Telegram gives you inline keyboards and free-text — that's it. This phase takes the approved flow from Phase 2 and turns the feature surface you extracted in Phase 1 into a navigable inline UX, then nails the three platform gotchas that will otherwise crash you in prod. Everything here is reusable: the four keyboard archetypes, the callback-data convention, the awaiting-state machine, and the interaction contract apply to any model-backed bot.

### The four keyboard archetypes

Almost every model bot reduces to four kinds of keyboard. Build one helper per archetype in a dedicated `keyboards.py` so the renderer and router stay in sync.

| Archetype | When to use | Layout rule | Example from the image-edit bot |
|---|---|---|---|
| **Feature grid** | Pick one of N capabilities the model exposes | 2 buttons/row (fits phone width, no horizontal scroll) | The 11 LoRA adapters as a grid; tapping one sets its sensible default prompt |
| **Direction submenu** | A feature has discrete variants | One row per variant + a Back button | Camera-angle / light-direction variants for features that have them |
| **+/- & toggle settings** | Numeric or boolean params the user may tune | `[ − ] [ short value ] [ + ]` for a **bare number**; a **labeled** value goes on its own full-width row with `[−] [+]` below (else Telegram truncates it — gotcha #30); a toggle button per boolean | `steps` and `cfg` stepper rows, a randomize-seed toggle, a "set seed" button |
| **Result actions** | After a result lands, offer next steps | Action verbs, 2/row | Repeat / Settings / New photo / Menu |

Plus a standalone **info button** (e.g. GPU/quota limits) — a read-only screen, no state change, reachable from the main menu.

Design rules learned the hard way:
- **2 buttons per row** for grids — single column wastes screen, 3+ truncate labels on narrow phones.
- **The current value lives *in* the keyboard** — the user reads state off the keyboard itself; you don't need a separate "current settings" message. **But mind the width:** a **bare short number** sits fine between the − and + buttons (`[−] [4] [+]`), whereas a **labeled** value (`🌡 Температура: 0.7`, `📏 Макс. токенов: 768`) squeezed into that middle slot gets **truncated by Telegram** — the number vanishes. Put a labeled value on its **own full-width row**, with `[−] [+]` on the row *below* it (gotcha #30).
- **Every navigable screen has a way back** (Back in submenus, Menu in result actions). Never strand the user.
- **One default-setting side effect per feature tap**: picking a feature should set its default prompt so the user can hit Generate immediately, while still being free to type a custom prompt instead.

### Callback-data scheme: one documented, compact convention

Every inline button carries a `callback_data` string, **hard-capped at 64 bytes by Telegram**. Overflow doesn't truncate gracefully — it fails. Adopt a terse, prefixed, colon-delimited scheme and **document it in one place** so `keyboards.py` (which emits it) and the router in `bot.py` (which parses it) never drift. This scheme is also a line item in the shared contract (Phase 6.1).

Convention: `<verb-or-namespace>[:<arg>[:<arg>]]`, prefixes ≤ 2 chars.

| callback_data | Meaning | Parsed args |
|---|---|---|
| `f:<key>` | pick feature | feature key |
| `d:<key>:<idx>` | pick direction within a feature | feature key, variant index |
| `gen` | generate | — |
| `again` | repeat last generation | — |
| `set` | open settings | — |
| `s:steps:-` / `s:steps:+` | decrement / increment steps | — |
| `s:cfg:-` / `s:cfg:+` | decrement / increment guidance | — |
| `s:rand` | toggle randomize-seed | — |
| `s:seed` | enter "set seed" mode | — |
| `menu` | back to main menu | — |
| `newphoto` | clear image, prompt for a new one | — |
| `limits` | show the quota/info screen | — |

Why this works and how to reuse it:
- **Prefix-route, don't string-match the whole thing.** The router switches on the prefix (`f`, `d`, `s`, …) then splits the rest. Adding a feature = adding a catalog entry, not a new handler.
- **Send keys, not labels or values.** `f:portrait` not `f:Portrait Lighting`; the human label lives in the catalog. Keeps you under 64 bytes and lets you rename labels without breaking buttons.
- **Indices for variants** (`d:<key>:0`) instead of variant names — shortest possible, and the catalog resolves index → variant.
- **Assert the limit in a test.** Build every keyboard the bot can produce and assert each button's `callback_data` ≤ 64 bytes (UTF-8). This caught real overruns before they shipped — make it part of your pre-deploy verification (Phase 6.4).

### The awaiting-state machine for free text

Inline buttons cover bounded choices; free text covers the unbounded ones (the prompt, a specific seed). Telegram delivers all of it through the *same* text handler, so you need a per-user **awaiting flag** to disambiguate what an incoming message *means*. (The session field that holds this flag is modeled in Phase 4.)

Store `session.awaiting ∈ {None, "prompt", "seed"}` on the per-user session. The text handler routes on it:

```python
def on_text(session, text):
    if session.awaiting == "seed":
        session.settings.set_seed(parse_int(text))   # clamps; see Phase 4
        session.awaiting = None
        return show_settings(session)
    elif session.awaiting == "prompt":
        session.prompt = text                          # custom prompt overrides feature default
        session.awaiting = None
        return show_ready_to_generate(session)
    else:
        # default: treat free text as a prompt if an image is loaded,
        # otherwise nudge the user
        if session.image_b64:
            session.prompt = text
            return show_ready_to_generate(session)
        return friendly_hint("Send a photo first, then describe the edit.")
```

Rules:
- **Buttons *arm* the state.** `s:seed` sets `awaiting = "seed"`; the prompt flow sets `awaiting = "prompt"`. The text handler only *consumes* it.
- **Always disarm after consuming** (`awaiting = None`) so the next message isn't misread.
- **Have a sane default** for unsolicited text (here: treat as a prompt when an image exists, else a friendly hint). Never silently drop a message.

### Interaction contract — the three must-know gotchas

These three caused real crashes / dead UI in prod. Bake them into your handler scaffolding for every bot. (They also appear, with root cause and evidence, in the consolidated Gotchas catalog near the end — here is the working fix.)

**1. You cannot `edit_message_text` on a media message.**
Once you've sent a photo or document (your result), its caption is *not* editable via `edit_message_text` — Telegram raises `BadRequest`. But menu re-renders naturally want to edit-in-place. Route **every** menu re-render through one helper that falls back to sending a fresh message:

```python
async def _show(query, text, reply_markup):
    try:
        await query.edit_message_text(text, reply_markup=reply_markup)
    except BadRequest:
        # message was media (photo/document) — can't edit its text; send fresh
        await query.message.reply_text(text, reply_markup=reply_markup)
```

Use `_show(...)` everywhere instead of calling `edit_message_text` directly. This was a genuine prod crash found in logs — the fix is cheap, the discipline (one chokepoint) is what makes it stick.

**2. A `CallbackQuery` can be answered exactly once.**
A second `query.answer()` on the same query raises `BadRequest`. Answer **once**, early, at the top of the callback handler — before any branching that might also try to answer.

**3. Always answer, always give a friendly fallback.**
- **Always call `query.answer()`** for every callback, even if you do nothing else — otherwise the user sees a spinner hang on the button.
- **Never dead-end.** If the user taps Generate with no image, or types before sending a photo, respond with a concrete hint ("send a photo first"), not silence and not a stack trace.

### Handler checklist (apply to every callback / message handler)
- [ ] Answer the callback query **once**, at the top.
- [ ] Re-render menus through `_show()` — never call `edit_message_text` directly (media-message trap).
- [ ] Parse `callback_data` by **prefix**, then split args; reject/ignore unknown prefixes gracefully.
- [ ] After consuming free text, **clear `session.awaiting`**.
- [ ] Provide a friendly fallback for every "can't do that yet" path (no image, no feature selected, bad seed).
- [ ] Keep every button's `callback_data` ≤ 64 bytes — assert it in a build-time test.
- [ ] Every screen has a Back / Menu exit; no dead ends.

---

## Phase 4 — State, Data Handling & Persistence

This phase is where a bot stops being a thin RPC shim and becomes a *product*: it remembers what each user is doing, adapts their input to whatever shape the backend actually wants (the shape you pinned in Phase 1), respects the chat platform's hard limits on the way out, and stays honest about a resource (quota) the provider refuses to expose via API (proven absent in Phase 1.4). Get this layer right and the bot survives restarts, bad input, and a silent backend.

There are five jobs here. Do all five — skipping any one produces a bot that *demos* well but *fails in prod*:

| Job | What it owns | The failure it prevents |
| --- | --- | --- |
| 1. Session modeling | Per-user dataclass + in-memory store | "Why did it use the other guy's photo?" |
| 2. Input adaptation | Telegram media → exact backend contract | Backend rejects upload / wrong type |
| 3. Output constraints | Caption limits, photo vs document | Telegram `BadRequest`, recompressed upscales |
| 4. Self-tracking | Count usage when no live API exists | Lying to the user about remaining quota |
| 5. Durable persistence | Atomic file on a mounted volume | Counters reset on every redeploy |

### 1. Model the session as a dataclass, store it in memory keyed by user_id

A bot is a state machine per user. Make that state an explicit `dataclass`, not a pile of `context.user_data["foo"]` dict keys — the dataclass gives you one place to see *everything a user's turn depends on*, and it's trivially serializable later if you ever need it.

```python
@dataclass
class Session:
    image_b64: str | None = None      # the current input, already in backend shape
    image_name: str | None = None
    feature_key: str | None = None    # which menu feature is selected
    prompt: str | None = None         # resolved prompt (feature default OR custom)
    settings: Settings = field(default_factory=Settings)
    awaiting: str | None = None       # {None, "prompt", "seed"} — the input router (Phase 3)
```

- [ ] **One dataclass per user holds the entire in-flight turn** — input, selection, resolved params, and the awaiting-flag. If a handler needs to know "what is this user doing right now," it reads one object.
- [ ] **`awaiting` is the free-text router** (the state machine from Phase 3). A `CallbackQuery` tells you exactly which button was pressed, but a plain text message is ambiguous: is it a custom prompt or a typed seed? The `awaiting` field disambiguates. Without it, every text message is a guess.
- [ ] **Settings live in their own dataclass with clamping baked in** — never trust a raw `+`/`-` button to stay in range. Each setter clamps; `__post_init__` clamps constructor values too:

```python
STEPS_MIN, STEPS_MAX = 1, 50
GUIDANCE_MIN, GUIDANCE_MAX = 1.0, 10.0
SEED_MIN, SEED_MAX = 0, 2147483647

def set_steps(self, value: int) -> int:
    self.steps = max(STEPS_MIN, min(STEPS_MAX, int(value)))
    return self.steps           # return the clamped value so the UI can re-render truthfully

def set_seed(self, value: int) -> int:
    self.seed = max(SEED_MIN, min(SEED_MAX, int(value)))
    self.randomize_seed = False  # typing a seed implies you want THAT seed
    return self.seed
```

  Keep the limits as **module-level constants** (`STEPS_MIN`, `GUIDANCE_MAX`, …) so the keyboard builder, the clamps, and the docs all read the same numbers. Drifting limits between UI and validation is a classic silent bug. (These same constants appear in the shared contract — Phase 6.1.)

- [ ] **`SessionStore` is a dict keyed by `user_id` with lazy create + explicit reset.** `get(user_id)` returns an empty `Session` on first touch; `reset(user_id)` replaces it wholesale. "New photo" and "Menu/start over" both just call `reset` — never hand-clear individual fields, you'll forget one.

```python
def get(self, user_id: int) -> Session:
    s = self._sessions.get(user_id)
    if s is None:
        s = Session(); self._sessions[user_id] = s
    return s
```

- [ ] **In-memory per-user state is fine; do NOT persist it.** The session is cheap to lose — a user just re-sends a photo. (Contrast with the *usage tracker* in job 5, which you must persist.) Keep the modules side-effect-free on import so they're unit-testable: `state.py` reads nothing from the environment and opens no connections at import time.

> **Why a singleton store and not `context.user_data`?** A single owned object makes "reset on new photo" one line and lets the session model evolve independently of PTB's storage. When you fork this bot, only the *fields* change — the store, the awaiting-router, and the clamping pattern carry over unchanged.

### 2. Adapt the input to the backend's *exact* contract — at the boundary

The single highest-leverage rule of this phase: **the session stores the input already in the backend's required shape, not in Telegram's shape.** Convert once, at the moment of receipt. Everything downstream (generate, repeat) then just passes `session.image_b64` through with zero re-conversion.

The backend here wanted a **base64 data-URI string**, not a file upload (discovered empirically in Phase 1 via `view_api()`). So on receipt:

```python
# Pick the LARGEST photo representation; for image-documents take the doc itself.
file_id = message.photo[-1].file_id if message.photo else message.document.file_id
tg_file = await context.bot.get_file(file_id)
raw = await tg_file.download_as_bytearray()

b64 = base64.b64encode(bytes(raw)).decode("ascii")
session.image_b64 = f"data:image/jpeg;base64,{b64}"   # the exact string /infer expects
```

- [ ] **Download the largest photo size** (`message.photo[-1]`) — Telegram sends an array of thumbnails; the last is full resolution. Grabbing `[0]` silently feeds the backend a postage stamp.
- [ ] **Accept both `filters.PHOTO` and `filters.Document.IMAGE`.** Users who care about quality send images *as documents* to dodge Telegram's compression. Handle both file_id sources in one handler.
- [ ] **Encode to the backend's literal contract.** Here that's `data:image/jpeg;base64,<…>` — Telegram photos are always JPEG, so the MIME is safe to hardcode. (If you accept image *documents*, they may be PNG; derive the MIME from the document's `mime_type` rather than hardcoding it for that path.) If your next backend wants a multipart upload or a URL, do *that* conversion here instead. The principle is invariant: **the boundary translates; the core never re-translates.**
- [ ] **Reset dependent state when the input changes.** A new photo invalidates the selected feature, prompt, and any pending input — clear `feature_key`, `prompt`, and `awaiting` in the same handler. Stale selections on fresh input are a confusing-result generator.
- [ ] **Wrap the download in try/except** and reply with a friendly error; network fetches from Telegram's CDN do fail.

### 3. Respect the platform's output limits — they will crash you otherwise

Telegram's send-side has two non-obvious hard rules. Both were found in prod logs, not docs.

| Constraint | The trap | The fix (do this every time) |
| --- | --- | --- |
| **Caption ≤ 1024 chars** | A long custom prompt blows the caption past 1024 → `BadRequest`, no result delivered. | Truncate the *prompt portion* (cap ~700) AND hard-cap the whole assembled caption at 1024 as a backstop. |
| **Photos get recompressed** | High-res / upscale results sent via `send_photo` come out blurry — Telegram re-encodes photos. | Send those as a **document** (`send_document`) so they arrive byte-for-byte. |

The caption assembly that worked — truncate the variable-length part first, then belt-and-suspenders cap the total:

```python
prompt_disp = session.prompt.strip()
if len(prompt_disp) > 700:
    prompt_disp = prompt_disp[:700] + "…"
caption = f"{feature.label}\n{prompt_disp}\nseed: {used_seed}\nsteps: {settings.steps}, cfg: {settings.guidance}"
if len(caption) > 1024:                 # backstop — never trust your own arithmetic
    caption = caption[:1023] + "…"
```

And the photo-vs-document branch keyed off a per-feature flag (`feature.is_upscale`) — so the *catalog* decides delivery mode, not scattered `if`s:

```python
if feature.is_upscale:
    await context.bot.send_document(chat_id=chat_id, document=bio, filename="result.png",
                                    caption=caption, reply_markup=result_keyboard())
else:
    await context.bot.send_photo(chat_id=chat_id, photo=bio,
                                 caption=caption, reply_markup=result_keyboard())
```

- [ ] Caption: truncate the **prompt** at ~700, then hard-cap the **whole caption** at 1024.
- [ ] Send high-res/upscale results as **documents**; normal results as photos.
- [ ] Let a per-feature flag pick the delivery mode, not ad-hoc conditionals.
- [ ] Wrap the send in try/except — if delivery fails *after* a successful (quota-costing!) generation, tell the user it generated but couldn't deliver, so they don't blindly retry and burn quota twice.

### 4. Self-track usage when the provider exposes no live API

We **proved** (Phase 1.4, by capturing all HTTP response headers during a live generation) that the provider returns **no** remaining-quota field — the real number appears *only* inside the quota-exceeded error text. So the bot must keep its own books.

The tracker model that worked:

- A **rolling 24h window** (`window_start`); on each record, if `now - window_start >= 24h`, reset the window (`_roll()`). This mirrors the provider's "resets 24h after first use" rule.
- A **generation counter** plus a **rough GPU-seconds estimate** = `wall_seconds × multiplier` (the Space is `xlarge`, so ×2). It's an upper bound, and labeled as an estimate in the UI — honesty over false precision.
- A cached `last_quota` report — the **authoritative** numbers, captured the one time the provider tells the truth (4b below).

```python
def record_generation(self, wall_seconds: float) -> None:
    self._roll()
    self.generations += 1
    self.est_gpu_seconds += max(0.0, wall_seconds) * XLARGE_MULTIPLIER  # rough upper bound
    self._save()
```

> Measure `wall_seconds` with `time.monotonic()` around the `await asyncio.to_thread(client.generate, …)` call — never wall-clock `time.time()`, which jumps under NTP.

**4b. Parse real numbers out of error text — match the ACTUAL wording.**

When the provider *does* leak the real remaining/reset, it's buried in free-form English. Two hard-won lessons:

1. **Match the wording the provider actually emits, character for character.** The reset clause read `"Try again in H:MM:SS"` — **not** `"retry in"`. An initial regex looking for `retry in` never matched, so the UI permanently said reset time "not reported" while the data was sitting right there. Capture *both* phrasings, and support both `H:MM:SS` and a bare `<n>s`:

```python
_RE_LEFT       = re.compile(r"(\d+(?:\.\d+)?)\s*s(?:econds)?\s*left", re.IGNORECASE)
_RE_RETRY_HMS  = re.compile(r"(?:try again|retry) in\s*(\d{1,2}:\d{2}(?::\d{2})?)", re.IGNORECASE)
_RE_RETRY_S    = re.compile(r"(?:try again|retry) in\s*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
```

2. **Parse `H:MM:SS` into seconds, then recompute the countdown from "now"** using the timestamp you captured when the error arrived (`at`). A stored absolute "2:39:03" is stale the instant you store it; what's durable is `(at, retry_seconds)`, and the live countdown is `(at + retry_seconds) - time.time()`:

```python
secs = 0
for part in match.group(1).split(":"):   # "2:39:03" → 9543
    secs = secs * 60 + int(part)
```

- [ ] Roll a 24h window matching the provider's reset rule.
- [ ] Estimate cost as `wall_seconds × multiplier`, label it an estimate, use `monotonic()`.
- [ ] Cache the authoritative numbers from the *one* moment the provider reveals them.
- [ ] Write regexes against the **literal observed wording** — verify against a real captured error string, not what you assume the message says.
- [ ] Store `(captured_at, retry_seconds)` and compute the live countdown on read.
- [ ] A quota *error* is also activity — open the window on it too, so the reset estimate works even with zero successful generations this session.

> The two regex/countdown traps above are the *same* facts the Backend phase relies on for honest quota wording — Phase 5 §5.6 references this section rather than repeating the parsing logic.

### 5. Durable persistence: atomic write to a mounted volume, fail soft

The usage tracker is the *one* piece of state that **must** survive restarts and redeploys — otherwise every deploy silently resets the user's quota countdown and counters, and the bot starts lying again. (The per-user sessions, by contrast, are disposable — don't bother persisting them.)

The pattern: JSON file on a **Dokploy named volume mounted at `/data`**, path overridable via `STATE_FILE` env, written **atomically**, loaded on construction, and **never allowed to crash the bot**. (The volume itself is provisioned in Phase 7.5.)

```python
STATE_FILE = os.environ.get("STATE_FILE", "/data/usage_state.json")

def _save(self) -> None:
    data = {...}
    try:
        os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, STATE_FILE)      # atomic: never leaves a half-written file
    except Exception as exc:             # persistence must NEVER take down the bot
        logger.warning("Could not save state (%s): %s", STATE_FILE, exc)

def _load(self) -> None:
    try:
        with open(STATE_FILE, encoding="utf-8") as fh:
            d = json.load(fh)
        # ... restore window_start, generations, est_gpu_seconds, last_quota ...
    except FileNotFoundError:
        logger.info("No state file yet — starting fresh")   # first run is NOT an error
    except Exception as exc:
        logger.warning("Could not load state: %s", exc)      # fall back to in-memory
```

- [ ] **Write tmp + `os.replace`**, never write the live file directly. `os.replace` is atomic on POSIX, so a crash mid-write can't corrupt the saved state — you either have the old file or the new one.
- [ ] **Mount a named volume at `/data`** (Phase 7.5) and confirm it's writable inside the container. Without a volume, the file lives in the container's ephemeral layer and dies on every redeploy — which defeats the entire point.
- [ ] **Override the path via `STATE_FILE`** so local dev (no `/data`) and prod use different locations without code changes.
- [ ] **Best-effort everywhere:** every IO path is wrapped; any failure logs a warning and falls back to in-memory operation. A bot that crashes because it couldn't write a stats file is worse than one with slightly stale stats.
- [ ] **Treat `FileNotFoundError` as normal** (first boot), distinct from real IO errors. Don't log a scary warning on a clean first start.
- [ ] **`_save()` on every mutating event** (each generation, each quota error). The state is tiny; the cost is negligible and you never lose more than the last action.

### 5b. (Space-side, optional) Persist generation history to a *private HF dataset*

The mounted-volume pattern above is the right tool when *you* control the host (Dokploy, your VPS). But sometimes the thing that must survive a restart lives on the **Space side** — e.g. a Space that hosts a gallery of every image it has ever generated. A HF Space **without paid persistent storage loses every file on restart**, and the `/data` named-volume trick isn't available there. The free, durable alternative is to **treat a private HF dataset repo as the persistence layer**.

The pattern: per generation, push the image plus a tiny caption sidecar to a **private** dataset repo (`huggingface_hub.create_commit`, one commit per image); on startup, pull the whole thing back into the gallery (`snapshot_download` of that dataset). It's free, it survives restarts, and the history is browsable and downloadable straight from the dataset page.

```python
from huggingface_hub import create_commit, CommitOperationAdd, snapshot_download

def persist_generation(img_bytes: bytes, caption: str, name: str) -> None:
    try:                                  # best-effort: a dataset hiccup must NOT crash the app
        create_commit(
            repo_id=HISTORY_DATASET, repo_type="dataset", token=HF_TOKEN,
            operations=[
                CommitOperationAdd(f"images/{name}.png", img_bytes),
                CommitOperationAdd(f"images/{name}.txt", caption.encode("utf-8")),
            ],
            commit_message=f"add {name}",
        )
    except Exception as exc:
        logger.warning("history push failed: %s", exc)

def load_history() -> str:
    try:
        return snapshot_download(repo_id=HISTORY_DATASET, repo_type="dataset", token=HF_TOKEN)
    except Exception as exc:
        logger.warning("history pull failed: %s", exc); return ""
```

- [ ] **Make the dataset *private*** — generation history is user content, not something to leak publicly. The same `HF_TOKEN` the Space already uses for its backends authorizes the commits.
- [ ] **One commit per image, with a sidecar** — the `.txt` (or `.json`) caption next to the image keeps the prompt/seed/settings attached, so the gallery can re-render labels after a `snapshot_download` with no separate index.
- [ ] **Pull on startup, push per generation** — `snapshot_download` once at boot rehydrates the gallery; each new generation appends a commit. The dataset *is* the source of truth across restarts.
- [ ] **Best-effort, exactly like the volume writer** — wrap both push and pull in try/except and log a warning. A dataset outage should degrade to "this session's images only," never take down the Space. Same principle as job 5: persistence failures are warnings, not crashes.

### Phase 4 done-check (run before claiming this layer works)

- [ ] Two users in parallel never see each other's image/prompt/settings.
- [ ] A typed message is correctly routed to prompt vs seed via `awaiting`.
- [ ] Settings can't escape their clamped ranges no matter how many times `+` is mashed.
- [ ] Input is stored in the backend's exact shape; generate/repeat re-use it with no re-conversion.
- [ ] A 900-char custom prompt produces a delivered result, not a `BadRequest`.
- [ ] An upscale arrives uncompressed (document); a normal result arrives as a photo.
- [ ] Counters and the quota countdown survive a full redeploy (verify by reading the volume file after a deploy).
- [ ] A real captured quota-error string parses into both remaining seconds and a reset countdown.
- [ ] Deleting/locking the state file degrades to in-memory — the bot keeps serving.

---

## Phase 5 — Backend Integration & Resilience

The bot's job is to be a thin, reliable, *honest* shell around a backend you don't control. Phase 1 gave you the verified API contract; this phase turns that contract into a client that survives version drift, never freezes the event loop, recreates itself on failure, and tells the user the truth when the provider says no. The recurring lesson: the failure modes are almost never "the model is wrong" — they're auth/quota/version/concurrency mismatches, and each one needs its own distinct user-facing message.

### 5.1 Discover the client/auth params — don't assume them

Client libraries rename their auth kwarg across versions, and getting it wrong doesn't error loudly — it silently downgrades you. With `gradio_client`, the auth kwarg in **2.5.0 is `token=`** (NOT `hf_token=`); older versions differ. **Why this matters concretely:** authenticating as the PRO account gets you **40 min of GPU budget/day**; falling through to *anonymous* gets you the **~2 min anon quota** — a 20× cliff that looks like "the bot mysteriously runs out almost immediately."

So: inspect the signature, then fall back through every known auth path before giving up.

```python
import inspect
def make_client(space_id, token):
    sig = inspect.signature(Client.__init__)
    if "token" in sig.parameters:        # gradio_client 2.5.0
        return Client(space_id, token=token)
    if "hf_token" in sig.parameters:     # older versions
        return Client(space_id, hf_token=token)
    # last resort: Authorization: Bearer header, then anonymous (logged WARN)
    try:
        return Client(space_id, headers={"Authorization": f"Bearer {token}"})
    except TypeError:
        log.warning("auth: falling back to ANONYMOUS — expect tiny quota")
        return Client(space_id)
```

- [ ] Pull the token from the secrets manager into env; never echo the value (see Phase 7.4).
- [ ] Inspect the client's constructor signature at runtime; pick the kwarg that actually exists.
- [ ] Order the fallback chain best→worst: `token=` → `hf_token=` → `Authorization: Bearer` header → anonymous.
- [ ] Log (WARN) which auth path won — anonymous must be visible in logs, not a silent surprise.
- [ ] Confirm with a live call that you got the *authenticated* quota, not the anon one.

**A PRIVATE backend needs a token just to be *reached* — this is not the same as the quota point above.** The 20× story is about a **public** Space, where anonymous still connects (you just inherit the tiny anon quota). A **private** Space is different in kind: `gradio_client` **cannot connect to it at all** without `token=` — there is no anonymous fallback, the handshake itself fails. So when the backend is private, the token is **mandatory, not best-effort**: the bot config must *require* it (validate at startup and refuse to run without it), and the auth-discovery fallback chain has no "anonymous" rung to land on — if every authed path fails, that's a hard config error, not a degraded mode.

### 5.2 Never block the event loop

`python-telegram-bot` runs on one asyncio loop. A blocking backend call (a `gradio_client.predict()` that takes 25–60s) will **freeze every other user's interaction** for the whole generation. Wrap every blocking backend call:

```python
result = await asyncio.to_thread(client.predict, image_b64, prompt, lora, ...)
```

- [ ] Every synchronous backend call goes through `asyncio.to_thread` (or equivalent executor).
- [ ] No file I/O, base64 encoding of large images, or `time.sleep` on the loop thread either.
- [ ] While the thread runs, the handler can still `answer()` the callback and post a "generating…" placeholder.

### 5.3 Lazy + cached client, reconnect once

Don't build the client at import time (it does a network handshake, and you want startup to be cheap and crash-free). Build it on first use, cache it, and on a connection error **recreate it once and retry** — handshakes go stale, Spaces restart.

```python
_client = None
def get_client():
    global _client
    if _client is None:
        _client = make_client(SPACE_ID, TOKEN)
    return _client

async def infer(...):
    try:
        return await _call(get_client(), ...)
    except ConnectionError:
        global _client; _client = None        # drop stale handle
        return await _call(get_client(), ...)  # recreate + retry ONCE
```

- [ ] Client created lazily, cached in a module global / singleton.
- [ ] On connection error: null the cache, recreate, retry **exactly once** (don't loop — a hard-down Space should surface as an error, not hang).

### 5.4 One bot, several backends — route by model, not by guesswork

A single bot can wrap **multiple backends with different signatures**, and the cached-singleton client above becomes a *small map* of clients rather than one global. Real example: two Spaces, both exposing a `/generate` endpoint, but with diverging arg lists —

- `generate(model_label, prompt, negative, steps, guidance, seed, randomize)`
- `generate(model_label, prompt, steps, guidance, seed, randomize)` — **no `negative` arg.**

If you build one universal argument list, you'll pass `negative` to a Space that doesn't take it (or drop it from one that needs it) and get a confusing positional-arg mismatch. The fix is to make the **model catalog the source of truth**: tag every model with the backend it lives on, then let the client pick the Space *by backend* and build the **per-backend argument list** (include or omit `negative`, reorder as that backend expects). Keep **one cached `gradio_client.Client` per backend** — the same lazy/cached/reconnect-once rules from 5.3 apply, just keyed by backend id.

```python
_clients: dict[str, Client] = {}            # keyed by backend id, not one global
def get_client(backend):
    if backend not in _clients:
        _clients[backend] = make_client(SPACES[backend], TOKEN)
    return _clients[backend]

def build_args(model):                       # per-backend arg list from the catalog
    a = [model.label, model.prompt]
    if model.backend == "sdxl":              # this backend takes a negative prompt
        a.append(model.negative)
    a += [model.steps, model.guidance, model.seed, model.randomize]
    return a
```

- [ ] Each model in the catalog is tagged with its backend; the client routes on that tag, never on a hardcoded default.
- [ ] The argument list is built **per backend** (include/omit/reorder), not shared across backends.
- [ ] One cached client per backend, each following the lazy + reconnect-once discipline of 5.3.

**Per-backend setting defaults.** When one bot spans model families, the *good* defaults diverge with the backend — so defaults must travel with the model, not be global constants. Concretely: **SDXL/Pony** models want **CFG ≈ 6 plus a negative prompt**; **FLUX** models want **guidance ≈ 3.5 and no negative** (passing a negative or a CFG of 6 to FLUX produces worse output, not an error — a silent quality bug). Either set sensible per-backend defaults the moment the user picks a model, or — at minimum — state the recommended values in the model-pick message so the user isn't flying blind.

- [ ] Defaults (CFG/guidance, negative on/off) are attached to the backend, applied on model selection.
- [ ] If you don't auto-apply them, the model-pick message states the recommended values explicitly.

### 5.5 Error taxonomy — one message per failure class

A single "something went wrong" is useless to the user *and* to you in the logs. Classify the failure and give each class a distinct, actionable message. Build it once in the client wrapper as typed exceptions (these named exceptions are part of the shared contract, Phase 6.1); the handler just maps type → text.

| Failure class | How it presents | Retryable? | User-facing message |
|---|---|---|---|
| **QuotaExceeded** | Provider error text with remaining/reset numbers | No — wait | Carry the *parsed* numbers: "~X s left, resets in H:MM:SS" (see 5.6) |
| **Transient connection** | `ConnectionError`, timeout, reset handshake | Yes — auto | "Hiccup talking to the backend — retrying…" then the one-shot retry from 5.3 |
| **Generic SpaceError** | Any other backend exception | No (fatal) | Friendly "the model failed on that one — try again or tweak the prompt" |

- [ ] Wrapper raises typed exceptions (`QuotaExceeded`, `BackendUnavailable`, `SpaceError`); handler never inspects raw strings.
- [ ] `QuotaExceeded` carries structured data (remaining seconds, reset countdown), not just a message.
- [ ] Transient errors are the only auto-retried class; quota and generic are not.

### 5.6 Honest quota wording — reservation vs exhaustion

The single most important integrity rule of this phase. The provider's quota error usually fires **when remaining < the reservation the task needs** — *not* at literal zero. If you parrot "quota exhausted," you'll lie to the user while there's still time on the clock.

**The parsing of the error text — matching the literal `"Try again in H:MM:SS"` wording, converting `H:MM:SS` → seconds, and recomputing the countdown from the captured timestamp — is owned by Phase 4 §4b.** This layer consumes the parsed `(remaining_seconds, reset_seconds)` that `QuotaExceeded` carries and decides only the *wording*:

- **The zero trap:** only say **"exhausted"** at true zero. Otherwise say you *can't start now*, and state the reservation size so the user understands *why*.

```python
def quota_message(remaining_s: int | None, reset_s: int | None) -> str:
    if remaining_s and remaining_s > 0:
        return (f"Can't start now: ~{remaining_s}s left, but the task reserves more "
                f"(xlarge ×2). Resets in {fmt(reset_s)}.")
    return f"Quota exhausted. Resets in {fmt(reset_s)}."
```

> Because the provider exposes **no live-quota API** (proven in Phase 1.4 by capturing all HTTP headers during a real generation — only generic rate-limit headers appear; the real numbers live *only* in the quota-exceeded error text), the bot also self-tracks usage over a rolling 24h window (Phase 4 §4). The parsed error numbers are the *authoritative* correction whenever they arrive.

- [ ] Take the already-parsed numbers from `QuotaExceeded` — do not re-parse strings here.
- [ ] "Exhausted" is reserved for true zero; otherwise "can't start now, task reserves X."
- [ ] State the reservation size in the message so the user understands *why* (e.g., "xlarge ×2").

### 5.7 Provider-side resource tuning — dynamic GPU duration

If you also control the backend (a Space you own/duplicated), tune its **GPU reservation** so generations can start with *less* remaining quota. Use a callable duration in the GPU decorator:

```python
@spaces.GPU(duration=lambda lora: 60 if lora not in LOADED else 25)
def infer(...): ...
```

- Return **60s** when the LoRA isn't loaded yet (the first-time adapter download happens *inside* the GPU function), **25s** when it's cached.
- **Why:** a smaller reservation lets a generation **start** when remaining quota is small — the tail of your daily quota isn't stranded. It does **NOT** reduce per-generation cost (you're billed by *actual* GPU time × the size multiplier, not by the reservation).

> Caveat from the field: a backend you don't fully own can betray you silently. A duplicated Space inherited the source image, so a copy that reported `RUNNING` was still running the *old* `xlarge` code — proven because its reservation logged "120s requested" (= 60 × 2). Lesson: **verify the backend is actually the build you think it is** (check the reservation math / a version marker in logs) before trusting it, and never ship a toggle that points at a backend that isn't confirmed working. (This is the same incident that motivates the "park, don't fake" principle — see the Engineering Principles and Phase 7.6.)

### 5.8 If you own the backend Space — per-type recipes (image SDXL · text chat)

When you build the backend Space yourself (not just wrap someone else's), two types recur, each with sharp edges found in the field. Both obey the **ZeroGPU load-fresh-per-call** rule (gotcha #19): `snapshot_download` every model's files at module scope, then load the *selected* model fresh inside the `@spaces.GPU` function (a forked worker won't keep a CUDA object across calls).

**(a) SDXL image generation — the bulletproof combo.** Naive SDXL on ZeroGPU fails two *silent* ways: the fp16 VAE overflows → a 200 + a full-frame **noise/black** PNG with no exception, and swapping in `DPMSolverMultistepScheduler(use_karras_sigmas=True)` → an **`IndexError`** in the denoise loop (`sigmas[step_index+1]`, which a `TypeError` guard won't catch). The combo that just works:

```python
from diffusers import AutoencoderKL, AutoPipelineForText2Image, EulerAncestralDiscreteScheduler
DTYPE = torch.float16
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=DTYPE)  # prefetch this repo too
pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, vae=vae, torch_dtype=DTYPE).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
```

- fp16 + the **fp16-fix VAE** kills the noise/black failure; **Euler a** is robust and never index-errors.
- **Look at the pixels** in your smoke test — a "successful" call can still return noise. In a **multi-model** Space, test **every** model: one can render fine while another silently noises with the *same* code (live: Pony Realism was clean, Lustify was pure noise — both needed the VAE-fix + Euler a) (gotchas #25, #26).

**(b) Text / LLM chat — transformers on ZeroGPU.** Expose a stable `/chat(model_label, message, history_json, system, temperature, max_tokens) -> reply` endpoint (history as a JSON string of OpenAI-style messages keeps the `gradio_client` contract simple). Two chat-template traps:

- **Qwen3 / "thinking" models leak chain-of-thought** — a `<think>…</think>` regex strip fails when `max_tokens` truncates before the closing tag. Kill it at the source: `tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)` — non-thinking templates ignore the unknown kwarg, so pass it unconditionally.
- **Gemma has no `system` role** — its template raises if you send one; fold the system text into the **first user message** instead.
- Decode **only the new tokens** (`out[0][input_len:]`), clamp `max_tokens`, and prefer **bf16** (Qwen3/Llama/Gemma are bf16-native; fp16 can NaN). (gotcha #28)
- **Robust think-strip:** even with `enable_thinking=False`, an abliterated model may still emit `<think>` and get cut off by `max_tokens`. Strip well-formed `<think>…</think>`, then also drop anything *before* a stray `</think>` and *after* an unmatched `<think>` — a single `<think>…</think>` regex misses the truncated case (gotcha #28).
- **Sampling is the biggest quality lever.** Generating with *temperature only* makes small / abliterated models ramble, loop, and feel "dumb." Always set `top_p=0.9`, `top_k=40`, `repetition_penalty=1.1` alongside temperature, and give `max_tokens` real headroom (≥768) so answers don't cut off mid-sentence (gotcha #31).

> Latency tradeoff: load-fresh-per-call means **every chat turn reloads the model** (~10s warm small model, ~100s cold 12B). Fine for a "try this model" stand; for interactive chat, flag it to the user — ZeroGPU is bursty by design, not an always-on serving tier. And the **first** call after the Space sleeps cold-boots (~2–4 min: container wake + module-scope prefetch/load) and can blow `gradio_client`'s default read timeout — poll `get_space_runtime` to `RUNNING` / pre-warm before a demo, and keep reconnect-once (5.3) so a warm retry recovers (gotcha #32).

### 5.9 Phase 5 exit checklist

- [ ] Auth kwarg discovered at runtime; confirmed running on the PRO (40-min) quota, not anonymous.
- [ ] If any backend is **private**, the token is required at startup (no anonymous fallback exists — connection itself fails without it).
- [ ] Every blocking backend call is off the event loop (`asyncio.to_thread`).
- [ ] Client is lazy, cached, and recreates-once on connection error — one cached client **per backend** if the bot wraps several.
- [ ] Models tagged with their backend; per-backend arg lists and setting defaults (e.g. SDXL CFG≈6 + negative vs FLUX guidance≈3.5, no negative) applied on selection.
- [ ] Three typed error classes, three distinct user messages; only transient is auto-retried.
- [ ] Quota messaging distinguishes "can't start (reservation)" from "exhausted (true zero)," consuming the countdown parsed in Phase 4 §4b.
- [ ] If you own the backend: dynamic GPU duration set, and the running build verified to be the one you intended.
- [ ] If you own an **SDXL** Space: fp16-fix VAE + Euler a (no noise/black, no scheduler IndexError); smoke test **inspected the actual pixels**, not just "no exception."
- [ ] If you own a **chat** Space: `enable_thinking=False` + Gemma system-fold handled; decode only new tokens; per-turn reload latency flagged to the user.

---

## Phase 6 — A Build Method That Produces Clean Code Fast (Contract → Parallel Modules → Adversarial Review → Verify)

By Phase 6 you have a validated API contract (Phase 1), an approved flow (Phase 2), and a UX/state design (Phases 3–4). Now you write the code. The trap is the obvious one: hand a single agent "build the bot" and it produces a 1,000-line `bot.py` with the LoRA strings copy-pasted three subtly-different ways, a `predict()` call that blocks the event loop, and a confident "done" with zero evidence. The method below avoids all of that. It is also *exactly* the shape of a Claude Code multi-agent run, so it maps 1:1 onto tools you already have.

**The four beats:** `Shared Contract` → `Parallel Modules` → `Adversarial Review` → `Fix`. Then one separate step that is non-negotiable: the **orchestrator** (you, or the lead agent — never the implementing agents) runs **real verification** before the word "done" is allowed.

### 6.1 Write the Shared Contract first (this is the whole game)

Parallel agents only produce coherent code if they share one source of truth. Write it *before* dispatching anyone. Without it, two agents invent two callback schemes and the router silently never fires half the buttons. With it, modules slot together on the first try — in this build `keyboards.py` emits `f:<key>` / `d:<key>:<idx>` / `s:steps:+` and the router in `bot.py` parses those exact strings, because both were copied from one document.

**The contract is the union of every prior phase's decisions, written down as a frozen interface surface** — it pulls the verbatim catalog from Phase 1.3, the callback scheme from Phase 3, the defaults/clamps from Phase 4, and the error taxonomy from Phase 5.4:

| Contract item | What to pin down | Example from this build | Source phase |
|---|---|---|---|
| Module list + public API | exact function signatures + return types per file | `SpaceClient.generate(image_b64, prompt, lora, seed, randomize, guidance, steps) -> tuple[bytes, int]` | this phase |
| Feature catalog | the EXACT provider strings, verbatim | 11 LoRA adapter strings + default prompts + direction variants | Phase 1.3 |
| Callback-data scheme | every token, each ≤ 64 bytes | `f:<key>`, `d:<key>:<idx>`, `gen`, `again`, `set`, `s:steps:±`, `s:cfg:±`, `s:rand`, `s:seed`, `menu`, `newphoto`, `limits` | Phase 3 |
| Generation defaults + limits | one place, with clamp ranges | `steps 1..50 (def 4)`, `guidance 1.0..10.0 (def 1.0)`, `seed 0..2147483647` | Phase 1.3 / 4 |
| Error taxonomy | the named exceptions agents must raise/catch | `QuotaExceeded` (carries `remaining_seconds`, `retry_seconds`) vs transient connection error vs generic `SpaceError` | Phase 5.4 |

Checklist — the contract is ready to dispatch on when:

- [ ] Every module has a one-line responsibility and a typed public API. Anything not in it is private.
- [ ] The catalog of magic strings (adapters, callback tokens) is written ONCE, verbatim — agents copy, never retype.
- [ ] Defaults and clamp limits are named constants, not literals scattered in code.
- [ ] The error taxonomy is named exceptions, so the backend agent and the handler agent agree on what crosses the boundary.
- [ ] A module's contract names its *dependencies*: e.g. "`space_client.py` imports only `gradio_client` + stdlib, never `telegram`." This is what makes it independently testable later.

### 6.2 Split into small, single-responsibility files — one agent each

Map the contract onto small files, then dispatch one agent per file **in parallel** (Claude Code: independent Task agents, or `superpowers:dispatching-parallel-agents`). Small files are not aesthetic — they are what makes parallelism safe (agents don't collide), review tractable (one lens per file), and rollback cheap (revert one module, not the bot). The split that worked:

| File | Single responsibility |
|---|---|
| `config.py` | env → typed `Config`; **no env reads at import** |
| `features.py` | the feature/LoRA catalog (the verbatim strings) |
| `state.py` | `Session`, `Settings` (with clamping), in-memory `SessionStore` |
| `space_client.py` | backend wrapper; gradio_client + stdlib only |
| `keyboards.py` | every inline keyboard; pure functions |
| `bot.py` | PTB handlers + the callback router |
| `usage.py` | usage tracking + atomic JSON persistence |
| packaging | `Dockerfile`, `requirements.txt`, `.env.example`, `.gitignore`, `README.md` |

Two rules that paid off and that you should put in every agent's brief:

- **No side effects at import.** Every module here states it (`config.py`: no env reads; `state.py`: no side effects). This is precisely what lets the orchestrator import-smoke all modules in a bare venv without a bot token or network — see 6.4.
- **Isolate the backend wrapper from the framework.** `space_client.py` has zero `telegram` imports, so it can be live-tested standalone against the real Space (and it was).

### 6.3 Adversarial review with multiple distinct lenses

Implementing agents are optimistic about their own code. So after the parallel build, run a **separate** review pass whose only job is to find problems — and run it with *distinct lenses*, because one reviewer chasing "is it correct" will skim right past cross-file drift. Use multiple review agents (Claude Code: `superpowers:requesting-code-review`, or parallel review Tasks), each with a single mandate:

| Lens | Hunts for | Real catches it should have made here |
|---|---|---|
| Interface consistency | does file A's call match file B's signature; callback tokens used == tokens defined | a button whose `callback_data` no router branch handles |
| Correctness / async | event-loop blocking, double-answer, edit-on-media, races | blocking `predict()` not wrapped in `asyncio.to_thread`; `query.answer()` called twice; `edit_message_text` on a photo message |
| Framework-API correctness **for the pinned version** | the library actually has this kwarg in THIS version | `gradio_client` auth kwarg is `token=` in 2.5.0, `hf_token=` in older — the code carries the fallback ladder precisely because review flagged version drift |

Then a **fix pass** that applies confirmed high/medium findings only — don't let the fixer gold-plate or "improve" things the review didn't flag. Receiving that feedback with rigor (verify each finding is real before acting; `superpowers:receiving-code-review`) keeps the fix pass honest both ways.

### 6.4 The orchestrator verifies for real — evidence before "done"

This is the hinge of the whole phase. **Implementing agents do not get to declare success. The orchestrator runs the checks and reads the output.** Claims without a command and its output are worthless (`superpowers:verification-before-completion`). In ascending order of cost, run all of them:

```text
1. compile       python -m py_compile *.py                  # syntax floor
2. import-smoke  venv + pip install -r req → import every module
                 # works ONLY because modules have no import side effects (6.2)
3. invariants    build EVERY keyboard, assert each callback_data ≤ 64 bytes;
                 assert settings clamp at their edges
                 # Telegram hard limit — a violation crashes at runtime, not build
4. live E2E      one real call through space_client.SpaceClient.generate(...)
                 # send a real photo b64, get bytes back, INSPECT the output image
```

Verification gate — do not ship until every box is checked with output you actually read:

- [ ] `py_compile` clean on all modules.
- [ ] Fresh venv installs `requirements.txt` and imports every module with no network and no secrets.
- [ ] Programmatically construct all keyboards and assert `len(callback_data.encode("utf-8")) <= 64` for every button.
- [ ] Settings clamp at their edges: `steps` stays in `1..50`, `guidance` in `1.0..10.0`, `seed` in `0..2147483647` (assert, don't eyeball).
- [ ] One **live end-to-end** generation through the real backend wrapper, authenticated as the PRO account — then **open the returned image** and confirm it's a real before/after edit, not a stack trace written to a file.

The live call is the one most people skip and the one that matters most: a green compile proves nothing about whether the contract you reverse-engineered in Phase 1 is actually right. Inspecting the output bytes is what turns "should work" into "works."

### 6.5 Why this is a Claude Code multi-agent workflow, verbatim

The four beats are not a metaphor for the agent loop — they *are* it:

| Beat | Claude Code mechanism |
|---|---|
| Shared Contract | the lead agent writes the spec/contract doc all subagents read (`superpowers:writing-plans`) |
| Parallel Modules | one Task per file, dispatched together (`superpowers:dispatching-parallel-agents`) |
| Adversarial Review | parallel review Tasks, one lens each (`superpowers:requesting-code-review`) |
| Fix + Verify | orchestrator-run commands, evidence required (`superpowers:verification-before-completion`) |

The discipline that makes it produce clean code fast — rather than fast garbage — is keeping the **orchestrator** as the one entity that owns the contract and owns verification. Agents implement and review; only the orchestrator decides it's done, and only after it has run the commands and looked at the output.

---

## Phase 7 — Deployment & Operations: shipping a long-polling bot 24/7 on a VPS

A model-backed Telegram bot is a long-lived process that talks *out* to two APIs (Telegram + your provider) and never accepts inbound traffic. That shape dictates everything below: no webhook, no domain, no open port — just one resilient container that polls, plus a deploy loop you can run a dozen times a day without fear.

This phase assumes the code from earlier phases is done. Here you package it, put it on a VPS via Dokploy, lock it down, give it durable storage (the persistence decision from Phase 4.5 and the hosting fork from Phase 2), and establish the commit→deploy→monitor→verify→rollback loop you'll reuse for every future bot.

### 7.1 Why long-polling in Docker (and when not to)

Default to **long-polling**, not webhooks, for a private/low-traffic model bot:

| | Long-polling (use this) | Webhook |
|---|---|---|
| Inbound network | none — bot calls `getUpdates` outbound | needs public HTTPS endpoint |
| Domain / TLS cert | not required | required |
| Open ports / Traefik routing | none | yes |
| Failure surface | one process, one outbound loop | reverse proxy + cert + routing |
| Cost of being wrong | restart the container | debug TLS/DNS/proxy chain |

A model bot is bottlenecked on GPU generation latency, not request throughput, so polling's overhead is irrelevant. Webhooks only earn their complexity at high message volume or when you genuinely can't make outbound long-lived connections. Reach for webhooks later, deliberately — never as the default.

**Container shape that worked** (`python-telegram-bot` v21, single-file entrypoint):

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "bot.py"]
```

Keep it this boring. `python:3.12-slim` + `pip install` + `CMD python bot.py` builds fast and has nothing to misconfigure. Ship `requirements.txt`, `.env.example`, `.gitignore`, and a `README` alongside it (the `.env.example` documents every var the bot reads; `.gitignore` keeps secrets and any `context-map-*/` notes out of the public repo).

### 7.2 The single-instance + stop-first rule (the one that bites everyone)

This is the highest-impact deployment gotcha for a polling bot on Docker Swarm, and it's invisible until prod logs show it.

**The failure:** Swarm's default update order is `start-first` — it starts the *new* task before stopping the *old* one for zero-downtime rollouts. For a polling bot that means **two containers briefly run at once, both call `getUpdates`**, and Telegram rejects the second:

```
Conflict: terminated by other getUpdates request; make sure that
only one bot instance is running
```

**The fix — two settings, both mandatory:**

- [ ] **Exactly 1 replica.** Polling must be single-instance; there is no "scale out" for a single bot token.
- [ ] **`updateConfig.Order = stop-first`** on the application. Stop the old task fully before starting the new one. This accepts a few seconds of downtime per deploy (fine for a bot) in exchange for never having two pollers.
- [ ] Set it **once** — it persists across deploys, so you won't re-trigger the Conflict on every redeploy.

> Rule of thumb: any single-token poller (Telegram, Discord gateway, IMAP idle, a queue consumer with an exclusive lock) needs `stop-first` + 1 replica on Swarm. Generalize this; don't relearn it per bot.

### 7.3 Dokploy provisioning (project → app → source → build → env)

Create the app via the Dokploy API/MCP in this order. Each step has a non-obvious requirement called out.

1. **Project + application.** One project per bot keeps env, volumes, and logs scoped together.
2. **Git source — pick the connection by repo visibility (see the decision tree below).** A **PUBLIC repo + `customGitUrl`** is the bulletproof default: no GitHub-App permission ambiguity, no token to manage on the Dokploy side, deploys "just work." It's safe because **the code has no secrets** — all secrets are injected as env vars (7.4), so the source itself is fine to expose.

   But "public" is a *choice*, not a law. A bot can legitimately need a **PRIVATE repo** — e.g. it wraps private/NSFW backends and you don't want the catalog, prompts, or backend Space IDs world-readable. `customGitUrl` **cannot clone a private repo** (Dokploy has no credentials for it), so the deploy fails at the clone step with no useful error. Decision tree:

   - **Public repo →** `customGitUrl`. Always prefer this when you can.
   - **Private repo, Dokploy GitHub App installed →** wire the source with `application.saveGithubProvider`, passing the `githubId` you read from `gitProvider.getAll`. The App carries the clone credentials, so a private repo deploys cleanly (this is the path that worked for the private-backend bot).
   - **Private repo, no GitHub App →** fall back to a PAT embedded in the `customGitUrl` (`https://<user>:<pat>@github.com/...`) — workable but now you're managing a token on the Dokploy side, so treat it with 7.4 hygiene.
   - **Last resort →** make the repo public (only valid once you've re-confirmed there are genuinely no secrets in it).

   Don't reach for the App or a PAT out of habit: if the repo *can* be public, `customGitUrl` is still the least-moving-parts option.
3. **Build type — `dockerfile`.** Note the API requires **all 7 build fields** present in the request even when most are defaults; send the full object or the call rejects.
4. **Environment — via `saveEnvironment`.** Inject secret *values* from the secrets manager directly into the request body. Never echo a value into the transcript or a log (see 7.4). The REST endpoint also requires `buildArgs`, `buildSecrets`, and `createEnvFile` to be **present** (e.g. `""`, `""`, `true`) — not just `{applicationId, env}` — or it 400s *"expected nonoptional, received undefined."* Tip: `application.update` can set `buildType`+`dockerfile`+`updateConfig.Order=stop-first`+`replicas` in **one** call, sidestepping `saveBuildType`'s many required-nullable fields (gotcha #29).
5. **`updateConfig.Order = stop-first`** (from 7.2) — set before the first deploy.
6. **A named volume mounted at `/data`** (`mounts.create type=volume`, see 7.5) — create before deploy so the first run already has durable storage.
7. **Trigger `application.deploy`.** Gotcha: `title` and `description` must be **non-null strings**, or the deploy call is rejected.

### 7.3a Run the whole Dokploy deploy as ONE process (not a chain of shell calls)

The provisioning sequence in 7.3 is half a dozen API calls whose outputs feed the next call (the project id feeds `application.create`, the app id feeds every `save*` and `deploy`, etc.). The tempting shape is a shell script: `bash dokploy-api.sh project.create | parse`, then `bash ... application.create | parse`, and so on — each call wrapped in a `$(...)` command-substitution and each parsed by another `python -c $(...)`. **Don't.** The agent's execution sandbox intermittently fails to fork subprocesses — `failed to change group ID: operation not permitted` — after roughly 3–4 forks inside a *single* shell invocation. A multi-step shell deploy blows that fork budget and dies partway through, leaving a half-created app you now have to clean up by hand.

**Robust shape: do the entire deploy in ONE process.** A single Python script using `httpx` straight against the Dokploy REST API, calling the endpoints in sequence and keeping the ids in memory between calls:

- Base URL `https://dokploy.moone.dev`, header `x-api-key: <key>`.
- Endpoints, in order: `/api/project.create` → `/api/application.create` → `/api/application.saveGithubProvider` (or the customGitUrl variant) → `/api/application.saveBuildType` → `/api/application.saveEnvironment` → `/api/application.update` (for `updateConfig.Order=stop-first`) → `/api/application.deploy`, then poll `/api/deployment.all`.
- **Build the env body with secret values held in memory only.** Read each secret from a file that was injected just beforehand, use it to construct the `saveEnvironment` body, then **delete the file immediately**. The value is never echoed to the transcript or interpolated on a shell command line.

One process means **one fork → no fork-budget problem**, the call graph is plain Python (no fragile `$()` parsing of JSON between steps), and secrets stay in process memory where 7.4's hygiene rules hold automatically. Reserve loose `bash dokploy-api.sh ...` one-shots for *single* read-only pokes (e.g. checking one deployment's status); never chain the build-out itself across shell calls.

### 7.3b Provisioning a ZeroGPU **backend** Space (when you build it, not just wrap it)

If the bot's backend is a Space *you* create, the reliable way to get **ZeroGPU** hardware is to **duplicate an existing working ZeroGPU Space**, then overwrite its code — a fresh `create_repo` Space defaults to CPU, and `duplicate_space` *requires* the hardware flavor named explicitly:

```python
api.duplicate_space(SRC_ZEROGPU_SPACE, "owner/new-space", private=True, hardware="zero-a10g")
api.add_space_secret(repo_id="owner/new-space", key="HF_TOKEN", value=tok)   # secrets are NOT copied
api.upload_folder(repo_id="owner/new-space", repo_type="space",
                  folder_path="./space", ignore_patterns=["__pycache__*"])   # overwrites app.py etc.
```

- `hardware="zero-a10g"` is the ZeroGPU API flavor (HF runs it on H200; `@spaces.GPU(size=…)` still picks large/xlarge). Omit it and `duplicate_space` 400s *"Hardware must be specified."* (gotcha #27)
- **Secrets don't copy** — re-set `HF_TOKEN` (and friends) on each new Space, or the module-scope prefetch/load of private or rate-limited model repos fails at request time.
- **Disk sizing:** a Space's container disk is ~50 GB and startup-cached model weights count against it, so **don't pack too many big models into one Space** — 4×(8–12 B) fp16 (~70 GB) won't fit; split across two Spaces (~32–40 GB each) and route across them (Phase 5.4). The build log's download progress shows the real footprint.
- After `upload_folder` the Space rebuilds; **poll `get_space_runtime(...).stage` to `RUNNING`** (not `RUNTIME_ERROR`), then **smoke-test with a real call and inspect the output** — "RUNNING" + "no exception" can still be noise (gotcha #25).

### 7.4 Secret hygiene (never let a value touch the transcript)

Treat every token as if a screen-share is recording. The discipline that worked:

- [ ] **Pull from a secrets manager into an env file** — never paste the literal token anywhere a human or log can see it.
- [ ] **Construct API bodies that reference the env var**, so the value is interpolated at call time and never typed out in your commands or chat.
- [ ] **Mask tokens** in any diagnostic/log output you produce (show `123...abc`, never the full string).
- [ ] **Allowlist at the app layer, not the repo.** The bot reads `ALLOWED_USER_IDS` (Telegram user_ids). Empty list ⇒ allow-all **with a logged warning** so an accidentally-open bot is loud, not silent. Auth lives in env, never in code.
- [ ] **Two distinct credentials**, both from the manager: the BotFather token (Telegram) and the provider/HF token (which determines your quota tier — authenticating as the PRO account is what gets the real quota, anonymous gets the tiny one).

### 7.5 Persistent state across restarts and redeploys

Container filesystems are ephemeral; a redeploy wipes them. Anything the bot must remember — usage counters, the rolling 24h GPU-minute estimate, the quota reset countdown (the tracker from Phase 4.5) — lives on a **Dokploy named volume** (`mounts.create type=volume`) mounted at **`/data`**.

- [ ] Create the volume as a **named** volume (survives redeploys; a bind mount or anonymous volume does not give the same guarantee).
- [ ] **Verify it's writable inside the running container** before trusting it — don't assume the mount took.
- [ ] Make the path overridable via a `STATE_FILE` env var so local iteration and prod can differ.
- [ ] Write **atomically** (the tmp-file + `os.replace` pattern from Phase 4.5).
- [ ] **Best-effort, never fatal:** any IO error falls back to in-memory state and logs; persistence failure must not crash the bot.

### 7.6 The deploy → monitor → verify → rollback loop

Run this *every* time you ship. It's the loop that let us iterate safely in prod.

```
commit  →  push  →  application.deploy  →  monitor status → done  →  verify RUNTIME logs  →  (rollback if bad)
          (retry)                          (background)
```

| Step | What to do | What "good" looks like |
|---|---|---|
| **Commit** | Small, single-purpose commits | one fix per deploy → trivial to bisect/rollback |
| **Push** | Push **with retries** | network/SSL flapped on us; don't treat a transient push failure as a code problem |
| **Deploy** | `application.deploy` (title/description non-null) | accepted |
| **Monitor** | Background-poll deployment status to `done` | build succeeds, status reaches done |
| **Verify runtime** | Read the **runtime logs**, not just build status | clean `getUpdates` 200s; **zero `Conflict`**; the auth/allowlist line printed; no tracebacks |
| **Rollback** | Keep the previous working build re-deployable; re-point/redeploy it | last known-good restored without a fresh build |

Verification checklist for the runtime logs:

- [ ] `getUpdates` returning **200** in a steady loop (the poller is alive)
- [ ] **Zero** `Conflict: terminated by other getUpdates request` (confirms 7.2 stuck)
- [ ] The startup auth line shows the **allowlist loaded** (and not the empty-list allow-all warning, unless intended)
- [ ] **No tracebacks** in the first minutes of runtime
- [ ] (If you touched the backend) one **live end-to-end generation** actually succeeds — build "done" is not proof the bot works
- [ ] **Tap through the real client.** Some defects never reach any log: a settings number truncated inside a button (gotcha #30), rambly "dumb" model output (gotcha #31), a cold-boot first-call timeout (gotcha #32). Build- and runtime-green ≠ good UX — a human using the actual bot is the only check that catches these. And when you fix such a bug, **retro-apply the fix to already-deployed Spaces/bots**, not just new ones.

> **Never ship to a backend that isn't actually working.** A deploy showing RUNNING can still be wrong — we caught a duplicated Space that reported RUNNING while silently executing the *old* code (proven by the reservation size in its logs; see Phase 5.6). Build/deploy status is necessary, not sufficient: confirm behavior from runtime evidence, and **park a feature rather than ship a broken toggle** when the backend can't be verified.

### 7.7 Operations quick-reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `Conflict: terminated by other getUpdates` | two pollers (start-first rollout or >1 replica) | `Order=stop-first` + exactly 1 replica |
| Counters/quota countdown reset after deploy | state on ephemeral FS | named volume at `/data`, atomic save |
| Bot replies to strangers | allowlist empty / not wired | set `ALLOWED_USER_IDS`; heed the allow-all warning |
| Tiny quota despite PRO account | bot authenticated anonymously | inject the PRO provider token via env; verify the won auth path in logs |
| Deploy call rejected | null title/description, or missing build fields | non-null strings; send all 7 build fields |
| `saveEnvironment` 400 "expected nonoptional" | REST endpoint needs `buildArgs`/`buildSecrets`/`createEnvFile` too | send them present (`""`,`""`,`true`); or use `application.update` for build+swarm config |
| New ZeroGPU Space won't take GPU / `duplicate_space` 400 | hardware not specified; fresh Space defaults to CPU | duplicate a working ZeroGPU Space with `hardware="zero-a10g"`, re-set secrets, `upload_folder` |
| First request after idle errors / times out | ZeroGPU cold boot (~2-4 min: wake + module-scope model load) | pre-warm / poll `get_space_runtime` to RUNNING; reconnect-once recovers a warm retry |
| Settings button shows no number / chat replies feel "dumb" | value truncated between ±; temperature-only sampling | value on its own full-width row (gotcha #30); add top_p/top_k/repetition_penalty (gotcha #31) |
| Push fails intermittently | flaky network/SSL | retry the push; it's not your code |

---

## Cross-cutting: the five engineering principles

These run underneath every phase. Each is the direct cause of a shipped fix or a saved hour.

1. **Verify empirically — don't guess.** If a fact will shape the architecture (input format, whether a quota API exists, VRAM fit), you must have *observed* it: a `view_api()`, a header dump, a byte count, a live call with the output inspected. We proved "there is no quota API" by capturing all response headers; we turned "will it OOM?" into a weight-file subtraction; we learned the contract from the wire, not the docs.

2. **Root-cause from prod logs.** Every shipped fix traced to a specific log line — the caption `BadRequest`, the `Conflict: terminated by other getUpdates` string, the `"retry in"` vs `"Try again in"` regex miss. If you can't point at the line that proves the cause, you're still guessing — go back to principle 1.

3. **Be honest when blocked — with evidence.** A blocked feature is a fact to report, not a thing to fake. When the true large-GPU backend wouldn't build (instant `exit 128`, log stuck on "Build Queued"), and a duplicated Space lied that it was RUNNING (proven by its "120s requested" = 60×2 reservation), we **parked the feature, said exactly why, and shipped no fake toggle.** A broken-but-present feature is worse than an honestly-absent one.

4. **Iterate safely in prod.** Small, reversible commits; monitor every deploy; keep a rollback artifact; never point the bot at a backend you haven't confirmed works. The deploy loop *is* the safety mechanism.

5. **Respect the platform's real constraints.** Most gotchas below aren't your bugs — they're the platform's rules: Telegram (1024-char captions, single poller, can't edit media text, answer-once), the provider (per-account GPU budget, size in a decorator, refusal at *reservation* not zero), Docker Swarm (`start-first` overlap → needs `stop-first` + 1 replica).

---

## Gotchas catalog

Every concrete trap from this build, with the root cause and the fix. Each appears **once** here as the canonical entry; the phases reference it. The "Why / evidence" column is what proved the cause — keep that habit.

| # | Area | Gotcha (symptom) | Root cause | Fix | Why / evidence |
|---|------|------------------|------------|-----|----------------|
| 1 | Telegram UX | `edit_message_text` on a photo/document message raises `BadRequest` | You can't edit the text of a media message | `_show(query, text, markup)` helper: try `edit_message_text`, on `BadRequest` send a fresh message; route all menu re-renders through it (Phase 3) | Real crash in prod logs |
| 2 | Telegram UX | Second `query.answer()` raises `BadRequest` | A `CallbackQuery` can be answered only once | Answer exactly once per callback, at the top (Phase 3) | Platform rule |
| 3 | Telegram output | Caption-too-long `BadRequest` | Telegram caption limit is 1024 chars | Truncate the prompt portion (cap ~700) and hard-cap the whole caption at 1024 (Phase 4 §3) | `BadRequest` in prod logs |
| 4 | Telegram output | Upscaled / high-res image comes back degraded | Sending as a *photo* makes Telegram recompress it | Send high-res results as a **document** (uncompressed) (Phase 4 §3) | Platform behavior |
| 5 | Quota parsing | UI wrongly says reset time "not reported" | Regex matched `"retry in"`, but the provider's wording is `"Try again in H:MM:SS"` | Match the **actual** wording, parse `H:MM:SS` → seconds, recompute the countdown from the captured "now" timestamp (Phase 4 §4b) | Reset time was *never* parsed until the keyword was corrected |
| 6 | Quota wording | Message says "exhausted" while quota remains | Quota error fires when remaining < the **reservation** the task needs, not at literal zero | Say "can't start now: ~X s left, the task reserves more (xlarge ×2)"; only say "exhausted" at true zero (Phase 5 §5.6) | Misleading wording found in testing |
| 7 | Quota API | No way to read remaining quota live | HF exposes **no** public quota API — only generic rate-limit headers | Self-track usage (count + GPU-minute estimate) over a rolling 24h window; parse exact numbers only from the quota-error text (Phase 1.4 / Phase 4 §4) | Proved by capturing **all** response headers during a live generation |
| 8 | Backend auth | Anonymous-tier tiny quota (~2 min) instead of PRO 40 min | Auth kwarg differs by version: gradio_client 2.5.0 wants `token=`, not `hf_token=`; if wrong, falls through to anonymous | Inspect the signature; fall back `token=` → `hf_token=` → `Authorization: Bearer` header → anonymous (Phase 5 §5.1) | Auth tier directly sets your quota ceiling |
| 9 | Async | Bot event loop stalls during generation | Blocking gradio_client calls run on the PTB event loop | Wrap every blocking backend call in `asyncio.to_thread` (Phase 5 §5.2) | python-telegram-bot is async |
| 10 | Backend resilience | A transient connection error kills the request | Cached client went stale | Lazily create + cache the client; on connection error, recreate **once** and retry (Phase 5 §5.3) | Spaces restart; handshakes go stale |
| 11 | GPU reservation | A generation refuses to start even with quota left | First-time LoRA download happens *inside* the GPU function, so a fixed reservation over-reserves | `@spaces.GPU(duration=callable)`: 60s when the LoRA isn't loaded, 25s when cached — a smaller reservation lets a job start with less remaining quota (Phase 5 §5.7) | Does **not** reduce per-gen cost (billed by actual time × multiplier), only the start threshold |
| 12 | Deployment | `Conflict: terminated by other getUpdates request` | Swarm default `start-first` update order briefly runs **two** containers → two pollers | Set app `updateConfig Order=stop-first`; keep exactly **1 replica** (Phase 7 §7.2) | Exact error string in prod logs |
| 13 | Deployment | "Live" backend silently runs old code | Duplicating a Space inherits the source image; a copy showing RUNNING still ran the old xlarge code | Don't trust the RUNNING badge — verify with a real call; here the "120s requested" reservation (60×2) proved it (Phase 5 §5.7 / Phase 7 §7.6) | Reservation math exposed the stale image |
| 14 | Deployment infra | Build fails instantly, no logs | Provider build infra stuck — `exit 128`, log only shows "Build Queued" | Park the feature, keep a rollback artifact, don't ship a fake toggle (Principle 3) | Live-streamed build log as evidence |
| 15 | Persistence | Limit countdown / counters reset on redeploy | State only in memory | Atomic save (write tmp → `os.replace`) of tracker JSON to a **named volume at `/data`**; best-effort fallback to in-memory; path overridable via `STATE_FILE` (Phase 4 §5) | Survives restarts **and** redeploys |
| 16 | State validation | Out-of-range settings reach the backend | No clamping | Clamp at the edge: steps 1..50, guidance 1.0..10.0, seed 0..2147483647, with module-level limit constants (Phase 4 §1) | — |
| 17 | Callback data | Inline button silently fails | Telegram caps `callback_data` at 64 bytes | Single documented compact scheme (`f:<key>`, `d:<key>:<idx>`, `s:steps:+`, …); orchestrator asserts every keyboard's `callback_data` ≤ 64 bytes (Phase 3 / Phase 6.4) | Caught by the build-time assertion, not in prod |
| 18 | Secret hygiene | Token leaks into transcript / logs | Echoing values during env setup or diagnostics | Inject secrets into an env file; reference env vars in API bodies; **mask tokens** in all log/diagnostic output (Phase 7 §7.4) | Standard, enforced throughout |
| 19 | ZeroGPU multi-model | Model dropdown is ignored — the Space always renders the same model/style | A module-level pipeline cache doesn't persist across calls: ZeroGPU forks a fresh worker per `@spaces.GPU` call, so mutations to the global cache are unreliable | Inside the `@spaces.GPU` function, load the **selected** model fresh **every** call; `snapshot_download` **all** model files at module level so the GPU window only loads from local disk, never the network. Correctness over the ~15-30 s switch-reload cost | Observed on a multi-model comparison stand — the dropdown selection never took effect with a cross-call cache |
| 20 | Gradio version skew | Space dies at startup with `RUNTIME_ERROR` (`Gallery.__init__() got an unexpected keyword argument 'show_download_button'`) | A newer gradio component kwarg doesn't exist in the gradio version that Space actually runs | Don't assume newer kwargs exist — verify each component kwarg against the **runtime** gradio version. (Here it was unnecessary anyway: the gallery's per-image download lightbox is on by default) | The exact `RUNTIME_ERROR` in the Space's startup log |
| 21 | Gated models | `403 Cannot access gated repo` when calling or prefetching a model | The model is `gated` and its license was never accepted for this account; the accept is a **UI consent form** with no clean programmatic path | A human must click "Agree and access" **once in the web UI** on the model page. On Discovery, check `model_info(repo).gated`; if gated, the error path must name the **exact model + page** to accept | `kpsss34/FHDR_Uncensored`, `black-forest-labs/FLUX.1-dev` are `gated=auto` — prefetch 403'd until accepted in the UI |
| 22 | Backend reachability | `gradio_client` can't even **connect** to the backend Space | The Space is **PRIVATE** — without a token you cannot reach it at all (distinct from gotcha #8, where a PUBLIC space still connects anonymously at a tiny quota) | When the backend is private, the auth token is **mandatory** — make the bot config **require** it; pass `token=` to reach the Space (Phase 5 §5.1) | A private Space refused the handshake entirely until `token=` was supplied |
| 23 | Deploy (private repo) | Dokploy can't clone a **PRIVATE** repo via `customGitUrl` | `customGitUrl` carries no credentials; a private repo needs auth to clone | Public repo → `customGitUrl` (no creds, bulletproof). Private repo → Dokploy **GitHub App** `application.saveGithubProvider` with the `githubId` from `gitProvider.getAll`; else a PAT embedded in `customGitUrl`; else make the repo public (Phase 7) | The GitHub-App path cloned bot #2's private repo where `customGitUrl` couldn't |
| 24 | Deploy (sandbox forks) | A multi-step shell deploy dies partway through with `failed to change group ID: operation not permitted` | The execution sandbox intermittently fails subprocess forks after ~3-4 forks in a **single** Bash invocation; a deploy made of many `$(bash dokploy-api.sh …)` + python-parse `$()`s blows that budget | Run the **entire** Dokploy deploy as **one** Python process — `httpx` straight to the REST API (`x-api-key` header; `/api/project.create`, `/api/application.create`, `.saveGithubProvider`, `.saveBuildType`, `.saveEnvironment`, `.update`, `.deploy`, `/api/deployment.all`), building the env body in memory, reading injected secret files and deleting them immediately. One process = one fork = no budget issue, and secrets are never echoed (Phase 7 §7.4) | The fork error reproduced reliably after 3-4 command-substitutions in one Bash call |
| 25 | SDXL image backend | Generation returns a **pure-noise / black** image with **no exception** | The SDXL fp16 VAE overflows and the latents decode to garbage — a *silent* failure (a 200 + a real-size PNG that's just noise) | **Bulletproof SDXL combo:** load `madebyollin/sdxl-vae-fp16-fix` (fp16) as the VAE + `EulerAncestralDiscreteScheduler` + fp16, and prefetch the VAE repo too. In a **multi-model** Space apply it to **every** model and **smoke-test each** — one model can render fine while another silently noises. Don't trust "no exception" — **look at the pixels** (Phase 5) | `lustify-sdxl` returned a full-frame RGB-noise PNG on bf16+default scheduler; in the multi-model stand Pony Realism rendered fine while Lustify was pure noise — both needed the VAE-fix + Euler a |
| 26 | SDXL scheduler | `IndexError: index N is out of bounds ... size N` **inside** the denoise loop | `DPMSolverMultistepScheduler(use_karras_sigmas=True)` indexes `self.sigmas[self.step_index + 1]` past the end on the final step on some diffusers builds; a `try/except TypeError` around the call does **not** catch it | Don't swap to DPM++ Karras blindly — use the model's default scheduler or `EulerAncestralDiscreteScheduler`. If you must guard a scheduler swap, catch `Exception`, not just `TypeError` (Phase 5) | Pulled the exact traceback from the Space run log (`scheduling_dpmsolver_multistep.py:961`) |
| 27 | ZeroGPU Space creation | `duplicate_space` 400s: "Hardware must be specified when duplicating a Space" | ZeroGPU isn't inferred on duplicate; the hardware flavor must be named | Pass `hardware="zero-a10g"` — the ZeroGPU API flavor (HF transparently runs it on H200; `@spaces.GPU(size=…)` still picks large/xlarge). **Duplicating an existing working ZeroGPU Space is the reliable way to provision a new ZeroGPU Space** (a fresh `create_repo` Space defaults to CPU); then `upload_folder` your code over it (Phase 7) | `duplicate_space(src, dst, private=True, hardware="zero-a10g")` stamped out 3 ZeroGPU Spaces in one run |
| 28 | LLM chat backend | Reply leaks raw chain-of-thought ("Okay, the user wants…"), or `apply_chat_template` errors on a `system` role | Qwen3 / thinking models emit a `<think>` block by default — a `<think>…</think>` regex strip fails when `max_tokens` truncates before the closing tag. Gemma's chat template rejects a `system` role outright | Suppress thinking at the source: `tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)` (non-thinking templates ignore the unknown kwarg). For Gemma-family, fold the system text into the **first user message** instead of a system turn. And if a stray `<think>` still survives and gets **truncated** by max_tokens (no closing tag), drop everything from the unmatched `<think>` onward — a `<think>…</think>` regex alone misses it (Phase 5) | Qwen3-abliterated leaked reasoning until `enable_thinking=False`; Gemma rejected the system role |
| 29 | Deploy (Dokploy fields) | `application.saveEnvironment` 400s: "expected nonoptional, received undefined" (`buildArgs` / `buildSecrets` / `createEnvFile`) | The REST endpoint requires those fields **present** (even empty / `true`), not just `{applicationId, env}` | Send `buildArgs:""`, `buildSecrets:""`, `createEnvFile:true` alongside the env. And `application.update` sets `buildType`+`dockerfile`+`updateConfigSwarm`(stop-first)+`replicas` in **one** call — sidestepping `saveBuildType`'s many required-nullable fields (Phase 7) | Each missing field surfaced in turn until the env saved (HTTP 200) |
| 30 | Telegram inline UX | A settings button's text is **truncated** — the value / number is invisible | Telegram shrinks a button that shares a row; a *labeled* value squeezed between `[−]`/`[+]` (or 3+ buttons per row) gets its text clipped | **Bare short numbers** (`[−] [4] [+]`) survive in-row, but a **labeled** value (`🌡 Температура: 0.7`) must go on its **own full-width row**, with the `[−] [+]` controls on the row *below* it (Phase 3) | "Настройки обрезаются, не видно цифр" — surfaced only when the user opened the real settings screen |
| 31 | LLM chat quality | Chat replies are rambly / incoherent / "dumb" — especially from small or abliterated models | Generating with **temperature only** (no nucleus / top-k / repetition control) lets the model loop and wander | Set real sampling: `top_p=0.9`, `top_k=40`, `repetition_penalty=1.1` alongside temperature — the single biggest coherence lever for ≤8–12 B abliterated models. Give max_tokens enough headroom (≥768) so answers don't cut off (Phase 5) | The user's models answered "тупо/стрёмно" until top_p/top_k/rep-penalty were added |
| 32 | ZeroGPU cold boot | The **first** call after the Space has been idle errors (read timeout / connection); a retry minutes later works | A slept ZeroGPU Space cold-boots on first request: container wake + the **module-scope model prefetch/load** can take ~2–4 min, exceeding `gradio_client`'s default read timeout | Poll `get_space_runtime(...).stage` to `RUNNING` before calling, **pre-warm** the Space before a demo, and keep the bot's reconnect-once (gotcha #10) so a warm retry recovers. Tell users the first message after idle is slow (Phase 5 / Phase 7) | Cold FLUX/text stands took ~4 min to wake; the first `gradio_client` call timed out, the next succeeded |

---

## New-bot quickstart

A numbered checklist to start the next model-backed bot from zero. Each step points at the phase that explains it. **Before step 1, have these prerequisites in hand:** access to the secrets manager, the target's owner/id (Space, model, or REST base URL), and — if deploying — a reachable Dokploy instance on a VPS. If any is missing, resolve it first; the chain below assumes all three.

1. **Authenticate, don't leak.** Pull the provider creds from the secrets manager; authenticate the CLI **without printing the value**. Confirm which account/tier you're on — it sets your quota. *(Phase 1.5, 7.4)*
2. **Classify the target.** Raw model vs. app/Space vs. REST. For a Space: `hf spaces info --format json`, then download and **read** `app.py` + `requirements.txt`. *(Phase 1.1)*
3. **Pin the I/O contract empirically.** `view_api()` (or OpenAPI); note the exact param *shapes* (e.g. base64 data-URI string, not a file); run **one live call** and **look at the artifact**. *(Phase 1.2)*
4. **Map capabilities and the cost model.** List tunable params + defaults, the verbatim feature catalog, what's fixed; find where the cost knob lives; write the quota math; **capture all headers** to prove whether a quota API exists. This produces your **contract sheet**. *(Phase 1.3–1.4)*
5. **Capture intent on paper.** Explore context, answer the 4 forks (input/extra-model · creds · hosting · audience), run the **safety gate**, and write the one-page UX flow. Get it approved before coding. *(Phase 2)*
6. **Write the shared contract.** Fold the contract sheet + the approved flow into: module list + typed public APIs, the verbatim string catalog, the callback-data scheme (each ≤ 64 bytes), defaults/clamp limits, the error taxonomy. *(Phase 6.1)*
7. **Build in parallel against the contract.** One small single-responsibility file per agent; no import side effects; backend wrapper free of `telegram` imports. *(Phase 6.2)*
8. **Adversarial review, then a narrow fix pass.** Three lenses: interface consistency, async/correctness, framework-API-for-this-version. Apply only confirmed findings. *(Phase 6.3)*
9. **Orchestrator verifies with evidence.** `py_compile` → bare-venv import-smoke → assert every `callback_data` ≤ 64 bytes and clamps hold → **one live E2E generation, inspect the output**. No "done" without output you read. *(Phase 6.4)*
10. **Package boringly.** `python:3.12-slim` Dockerfile, `requirements.txt`, `.env.example`, `.gitignore`, `README`. *(Phase 7.1)*
11. **Provision on Dokploy.** Public repo via `customGitUrl`, `dockerfile` build (all 7 fields), env via `saveEnvironment`, `updateConfig.Order=stop-first` + **1 replica**, named volume at `/data`, then `application.deploy` (non-null title/description). *(Phase 7.2–7.5)*
12. **Run the deploy loop.** commit → push (with retries) → deploy → background-monitor to done → **verify runtime logs** (200 `getUpdates`, zero `Conflict`, allowlist line, no tracebacks) → keep a rollback artifact. *(Phase 7.6)*

### Recommended file / module layout

```
bot/
├── bot.py            # entrypoint: PTB app, handler registration, the callback router
├── config.py         # env → typed Config; NO env reads at import time
├── features.py       # the feature/capability catalog — verbatim backend strings
├── keyboards.py      # all inline keyboards as pure functions (emits the callback scheme)
├── state.py          # Session + Settings dataclasses (clamping), in-memory SessionStore
├── space_client.py   # backend wrapper: gradio_client + stdlib ONLY, no telegram imports
├── usage.py          # usage tracker + atomic JSON persistence (writes /data, STATE_FILE)
│
├── Dockerfile        # python:3.12-slim, pip install, CMD ["python","bot.py"]
├── requirements.txt  # python-telegram-bot==21.*, gradio_client, ...
├── .env.example      # documents EVERY env var: BOT_TOKEN, HF_TOKEN, ALLOWED_USER_IDS, STATE_FILE
├── .gitignore        # .env, __pycache__, context-map-*/ — keep secrets out of the public repo
└── README.md         # what it is, the env vars, how to run locally vs. deploy
```

**The shape to internalize:** small single-responsibility modules around one frozen contract; the backend wrapper isolated so it's testable alone; secrets only in env; durable state on a mounted volume; a single-instance poller that stops-first on every deploy; and nothing called "done" until a live call's output has been looked at.
