"""
YGO Deck Editor — GUI for creating and editing YDK deck files.

Usage:
    python scripts/deck_editor.py

Optional dependency for card image previews:
    pip install Pillow

Card names and ban/MD status are loaded from:
    data/card_names.json  (run scripts/build_card_db.py first, or click Update Cards)
    data/card_info.json   (same script)

Ban legend shown in toolbar:
    [F]  Forbidden in Master Duel
    [L]  Limited (1 copy)
    [SL] Semi-Limited (2 copies)
    [!]  Card not yet available in Master Duel
"""

from __future__ import annotations

import io
import json
import threading
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image as PILImage, ImageTk  # type: ignore
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ─── URLs ────────────────────────────────────────────────────────────────────
YGOPRODECK_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php?misc=yes"
IMAGE_BASE_URL = "https://images.ygoprodeck.com/images/cards_small"

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CARD_DB_PATH   = ROOT / "data" / "card_names.json"
CARD_INFO_PATH = ROOT / "data" / "card_info.json"
ENGINES_DIR    = ROOT / "data" / "engines"

# ─── Deck rules ──────────────────────────────────────────────────────────────
MAIN_MIN, MAIN_MAX = 40, 60
EXTRA_MAX = 15
SIDE_MAX  = 15
MAX_COPIES = 3

# ─── Colors ──────────────────────────────────────────────────────────────────
COLOR_OK        = "#2ecc71"
COLOR_WARN      = "#e67e22"
COLOR_ERROR     = "#e74c3c"
COLOR_BG        = "#1a1a2e"
COLOR_PANEL     = "#16213e"
COLOR_ACCENT    = "#0f3460"
COLOR_TEXT      = "#e0e0e0"
COLOR_HIGHLIGHT = "#e94560"
COLOR_ENTRY     = "#0f3460"
COLOR_FORBIDDEN = "#e74c3c"
COLOR_LIMITED   = "#f39c12"
COLOR_SEMI      = "#f1c40f"
COLOR_NOT_MD    = "#666666"


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_card_db() -> dict[int, str]:
    if not CARD_DB_PATH.exists():
        return {}
    with open(CARD_DB_PATH, encoding="utf-8") as f:
        raw: dict[str, str] = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_card_info() -> dict[str, dict]:
    """Load ban/MD status per card. Returns {} if file missing."""
    if not CARD_INFO_PATH.exists():
        return {}
    with open(CARD_INFO_PATH, encoding="utf-8") as f:
        return json.load(f)


def parse_ydk(path: Path) -> tuple[list[int], list[int], list[int]]:
    main, extra, side = [], [], []
    section = "main"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#created"):
                continue
            if line == "#main":
                section = "main"
            elif line == "#extra":
                section = "extra"
            elif line == "!side":
                section = "side"
            else:
                try:
                    code = int(line)
                    if section == "main":
                        main.append(code)
                    elif section == "extra":
                        extra.append(code)
                    else:
                        side.append(code)
                except ValueError:
                    pass
    return main, extra, side


def write_ydk(path: Path, main: list[int], extra: list[int], side: list[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("#created by YGO Deck Editor\n")
        f.write("#main\n")
        for code in main:
            f.write(f"{code}\n")
        f.write("#extra\n")
        for code in extra:
            f.write(f"{code}\n")
        f.write("!side\n")
        for code in side:
            f.write(f"{code}\n")


def card_display_info(card_info: dict[str, dict], code: int) -> tuple[str, str]:
    """
    Returns (prefix_text, text_color) for a card code.
    prefix_text: "[F]", "[L]", "[SL]", "[!]", or ""
    text_color:  one of the COLOR_* constants
    """
    entry = card_info.get(str(code), {})
    in_md: bool = entry.get("in_md", True)   # default True if info missing
    # ban_ocg is used as the closest public approximation of the MD banlist
    ban = entry.get("ban_ocg")
    if not in_md:
        return "[!]", COLOR_NOT_MD
    if ban in ("Forbidden", "Banned"):
        return "[F]", COLOR_FORBIDDEN
    if ban == "Limited":
        return "[L]", COLOR_LIMITED
    if ban == "Semi-Limited":
        return "[SL]", COLOR_SEMI
    return "", COLOR_TEXT


def is_extra_deck_card(card_info: dict[str, dict], code: int) -> bool:
    """True for Fusion / Synchro / Xyz / Link monsters (belong in Extra Deck)."""
    return card_info.get(str(code), {}).get("is_extra", False)


# ─── Shared image fetch ───────────────────────────────────────────────────────

def _fetch_card_image(code: int, on_done: "callable[[], None] | None" = None) -> None:
    """Download card image into ImageTooltip._cache. Runs in a background thread."""
    url = f"{IMAGE_BASE_URL}/{code}.jpg"
    try:
        if not PIL_AVAILABLE:
            raise ImportError("Pillow not installed")
        req = urllib.request.Request(url, headers={"User-Agent": "ygo-meta-ai/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        img = PILImage.open(io.BytesIO(data))
        img.thumbnail((177, 254), PILImage.LANCZOS)
        ImageTooltip._cache[code] = ImageTk.PhotoImage(img)
    except ImportError:
        ImageTooltip._cache[code] = None   # Pillow not available
    except Exception:
        ImageTooltip._cache[code] = None   # network / decode error
    finally:
        ImageTooltip._pending.discard(code)
    if on_done:
        on_done()


# ─── Image tooltip ────────────────────────────────────────────────────────────

class ImageTooltip:
    """Card image popup shown on mouse hover over a widget."""

    _cache: dict[int, object] = {}   # code → PhotoImage | None (None = unavailable)
    _pending: set[int] = set()       # codes currently being fetched

    def __init__(self, widget: tk.Widget, code: int) -> None:
        self._widget = widget
        self._code = code
        self._tip: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")

    def _on_enter(self, _: tk.Event) -> None:
        self._after_id = self._widget.after(400, self._show)

    def _on_leave(self, _: tk.Event) -> None:
        if self._after_id:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self) -> None:
        self._after_id = None
        if self._tip and self._tip.winfo_exists():
            return
        try:
            rx = self._widget.winfo_rootx() + self._widget.winfo_width() + 6
            ry = self._widget.winfo_rooty()
        except tk.TclError:
            return
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{rx}+{ry}")
        self._tip.configure(bg="#000000")

        if self._code in ImageTooltip._cache:
            self._render()
        else:
            tk.Label(self._tip, text="Loading…", bg="#1a1a2e", fg="#888888",
                     padx=20, pady=16, font=("Segoe UI", 9)).pack()
            if self._code not in ImageTooltip._pending:
                ImageTooltip._pending.add(self._code)
                threading.Thread(
                    target=_fetch_card_image,
                    args=(self._code, lambda: self._schedule_render()),
                    daemon=True,
                ).start()

    def _schedule_render(self) -> None:
        if self._tip and self._tip.winfo_exists():
            try:
                self._widget.after(0, self._render)
            except tk.TclError:
                pass

    def _render(self) -> None:
        if not (self._tip and self._tip.winfo_exists()):
            return
        for w in self._tip.winfo_children():
            w.destroy()
        img = ImageTooltip._cache.get(self._code)
        if img:
            lbl = tk.Label(self._tip, image=img, bg="#000000", bd=1, relief="solid")
            lbl.image = img   # prevent GC
            lbl.pack()
        else:
            msg = (
                "No image available"
                if PIL_AVAILABLE
                else "Install Pillow for card images:\npip install Pillow"
            )
            tk.Label(self._tip, text=msg, bg="#1a1a2e", fg="#888888",
                     padx=20, pady=16, font=("Segoe UI", 9)).pack()

    def _hide(self) -> None:
        if self._tip:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


# ─── Scrollable card image grid ────────────────────────────────────────────────────

class CardScrollList(tk.Frame):
    """
    Deck section display: scrollable grid of card images.
    Each cell shows: card image + copy count badge + small name text.
    Left-click removes one copy; right-click removes all copies.
    """

    THUMB_W, THUMB_H = 68, 100
    CELL_W = 82

    _grid_cache: dict[int, object] = {}   # code -> grid-sized PhotoImage | None
    _grid_pending: set[int] = set()
    _grid_photos: list = []               # prevent GC of PhotoImages

    def __init__(
        self,
        parent: tk.Widget,
        card_db: dict[int, str],
        card_info: dict[str, dict],
        on_remove_one: "callable[[int], None]",
        on_remove_all: "callable[[int], None]",
        **kw,
    ) -> None:
        bg = kw.pop("bg", COLOR_PANEL)
        super().__init__(parent, bg=bg, **kw)
        self._card_db = card_db
        self.card_info = card_info          # public: replaced by DeckEditor after refresh
        self._on_remove_one = on_remove_one
        self._on_remove_all = on_remove_all
        self._tooltips: list[ImageTooltip] = []
        self._tiles: list[tuple[tk.Frame, int]] = []
        self._cols: int = 0

        self._canvas = tk.Canvas(self, bg=bg, bd=0, highlightthickness=0)
        self._sb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._sb.set)
        self._sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(self._canvas, bg=bg)
        self._win_id = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>", self._on_inner_cfg)
        self._canvas.bind("<Configure>", self._on_canvas_cfg)
        for w in (self._canvas, self._inner):
            w.bind("<MouseWheel>", self._on_wheel)

    def _on_inner_cfg(self, _: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_cfg(self, event: tk.Event) -> None:
        self._canvas.itemconfig(self._win_id, width=event.width)
        new_cols = max(1, event.width // self.CELL_W)
        if new_cols != self._cols:
            self._cols = new_cols
            self._reflow()

    def _on_wheel(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def populate(self, deck: list[int]) -> None:
        self._tooltips.clear()
        for w in self._inner.winfo_children():
            w.destroy()
        self._tiles.clear()

        # Group preserving first-occurrence order
        seen: dict[int, int] = {}
        order: list[int] = []
        for code in deck:
            if code not in seen:
                seen[code] = 0
                order.append(code)
            seen[code] += 1

        for code in order:
            tile = self._make_tile(code, seen[code])
            self._tiles.append((tile, code))

        self._cols = 0
        self._inner.update_idletasks()
        w = self._canvas.winfo_width()
        if w > 1:
            self._cols = max(1, w // self.CELL_W)
        self._reflow()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

        # Prefetch images for all cards in the deck
        self._prefetch(order)

    def _make_tile(self, code: int, copies: int) -> tk.Frame:
        prefix, color = card_display_info(self.card_info, code)

        tile = tk.Frame(self._inner, bg=COLOR_PANEL)

        # Image container (fixed size)
        img_frame = tk.Frame(tile, bg="#222222", width=self.THUMB_W, height=self.THUMB_H)
        img_frame.pack_propagate(False)
        img_frame.pack(side="top", padx=5, pady=(3, 0))

        img_lbl = tk.Label(img_frame, bg="#222222", text="", fg="#555")
        img_lbl.pack(fill="both", expand=True)

        # Show cached image immediately if available
        cached = CardScrollList._grid_cache.get(code)
        if cached:
            img_lbl.configure(image=cached)
            img_lbl.image = cached

        # Copy count badge (top-right of image)
        if copies > 1:
            badge = tk.Label(img_frame, text=f"x{copies}", bg=COLOR_HIGHLIGHT, fg="white",
                             font=("Consolas", 7, "bold"), padx=2, pady=0)
            badge.place(relx=1.0, rely=0.0, anchor="ne")
            badge.bind("<Button-1>", lambda e, c=code: self._on_remove_one(c))
            badge.bind("<Button-3>", lambda e, c=code: self._on_remove_all(c))

        # Card name (small text below image)
        name = self._card_db.get(code, f"#{code}")
        short = (name[:10] + "…") if len(name) > 11 else name
        name_lbl = tk.Label(tile, text=short, bg=COLOR_PANEL, fg=color,
                            font=("Segoe UI", 7), anchor="center")
        name_lbl.pack(side="top", fill="x", padx=1)

        # Tooltip on hover (shows full-size image + full name)
        tip = ImageTooltip(tile, code)
        self._tooltips.append(tip)

        # Click bindings: left = remove one, right = remove all
        for w in (tile, img_frame, img_lbl, name_lbl):
            w.bind("<Button-1>", lambda e, c=code: self._on_remove_one(c))
            w.bind("<Button-3>", lambda e, c=code: self._on_remove_all(c))
            w.bind("<MouseWheel>", self._on_wheel)

        tile._img_lbl = img_lbl  # reference for async image update
        return tile

    def _reflow(self) -> None:
        if not self._tiles:
            return
        cols = max(1, self._cols) if self._cols else 1
        for i, (tile, _) in enumerate(self._tiles):
            r, c = divmod(i, cols)
            tile.grid(row=r, column=c, padx=1, pady=1, sticky="nw")
        self._inner.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _prefetch(self, codes: list[int]) -> None:
        for code in codes:
            if code in CardScrollList._grid_cache or code in CardScrollList._grid_pending:
                continue
            CardScrollList._grid_pending.add(code)
            threading.Thread(
                target=self._fetch_grid_image, args=(code,), daemon=True
            ).start()

    def _fetch_grid_image(self, code: int) -> None:
        url = f"{IMAGE_BASE_URL}/{code}.jpg"
        try:
            if not PIL_AVAILABLE:
                raise ImportError("Pillow not installed")
            req = urllib.request.Request(url, headers={"User-Agent": "ygo-meta-ai/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            pil_img = PILImage.open(io.BytesIO(data))
            grid_copy = pil_img.copy()
            grid_copy.thumbnail((self.THUMB_W, self.THUMB_H), PILImage.LANCZOS)
            tip_copy = pil_img.copy()
            tip_copy.thumbnail((177, 254), PILImage.LANCZOS)
            try:
                self._canvas.after(0, lambda: self._on_img_ready(code, grid_copy, tip_copy))
            except tk.TclError:
                CardScrollList._grid_pending.discard(code)
        except Exception:
            CardScrollList._grid_cache[code] = None
            CardScrollList._grid_pending.discard(code)

    def _on_img_ready(self, code: int, grid_pil, tip_pil) -> None:
        try:
            grid_photo = ImageTk.PhotoImage(grid_pil)
            CardScrollList._grid_cache[code] = grid_photo
            CardScrollList._grid_photos.append(grid_photo)
            if code not in ImageTooltip._cache:
                tip_photo = ImageTk.PhotoImage(tip_pil)
                ImageTooltip._cache[code] = tip_photo
                CardScrollList._grid_photos.append(tip_photo)
        except Exception:
            CardScrollList._grid_cache[code] = None
        finally:
            CardScrollList._grid_pending.discard(code)

        # Update tile image
        for tile, c in self._tiles:
            if c == code:
                img = CardScrollList._grid_cache.get(code)
                if img:
                    tile._img_lbl.configure(image=img, text="")
                    tile._img_lbl.image = img
                break


# ─── Main application ─────────────────────────────────────────────────────────

class DeckEditor(tk.Tk):

    def __init__(self) -> None:
        super().__init__()
        self.title("YGO Deck Editor")
        self.geometry("1400x820")
        self.minsize(1060, 680)
        self.configure(bg=COLOR_BG)

        self._card_db: dict[int, str] = load_card_db()
        self._card_info: dict[str, dict] = load_card_info()
        self._all_cards_by_name: list[tuple[str, int]] = sorted(
            ((v, k) for k, v in self._card_db.items()), key=lambda x: x[0].lower()
        )

        self._main:  list[int] = []
        self._extra: list[int] = []
        self._side:  list[int] = []
        self._current_file: Path | None = None

        # Hover tooltip state for search Listbox
        self._lb_tip: tk.Toplevel | None = None
        self._lb_tip_code: int | None = None
        self._lb_tip_after: str | None = None

        self._build_ui()
        self._refresh_all()

        if not self._card_db:
            messagebox.showwarning(
                "Card database missing",
                "data/card_names.json not found.\n"
                "Run:  python scripts/build_card_db.py\n\n"
                "You can still load/edit YDK files; card names won't show until the DB is built.",
            )

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("TFrame", background=COLOR_BG)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
        style.configure("TButton", background=COLOR_ACCENT, foreground=COLOR_TEXT,
                        padding=6, relief="flat")
        style.map("TButton",
                  background=[("active", COLOR_HIGHLIGHT), ("pressed", COLOR_HIGHLIGHT)])
        style.configure("Accent.TButton", background=COLOR_HIGHLIGHT, foreground="white", padding=6)
        style.configure("TEntry", fieldbackground=COLOR_ENTRY, foreground=COLOR_TEXT,
                        insertcolor=COLOR_TEXT)
        style.configure("TScrollbar", background=COLOR_ACCENT, troughcolor=COLOR_PANEL,
                        arrowcolor=COLOR_TEXT)

        # ── Toolbar ──────────────────────────────────────────────────────────
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=8, pady=(8, 0))

        ttk.Button(toolbar, text="📂 Load",
                   command=self._load_ydk).pack(side="left", padx=2)
        ttk.Button(toolbar, text="💾 Save",
                   command=self._save_ydk,
                   style="Accent.TButton").pack(side="left", padx=2)
        ttk.Button(toolbar, text="💾 Save As",
                   command=self._save_ydk_as).pack(side="left", padx=2)
        ttk.Button(toolbar, text="🗑 Clear",
                   command=self._clear_deck).pack(side="left", padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", padx=8, fill="y")

        self._refresh_btn = ttk.Button(toolbar, text="🔄 Update Cards",
                                       command=self._refresh_cards_async)
        self._refresh_btn.pack(side="left", padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", padx=8, fill="y")

        # Legend (ban status is OCG — best available approximation of MD banlist)
        for text, color in [
            ("[F] Forbidden",     COLOR_FORBIDDEN),
            ("[L] Limited",       COLOR_LIMITED),
            ("[SL] Semi-Lim",     COLOR_SEMI),
            ("[!] Not in MD",     COLOR_NOT_MD),
        ]:
            tk.Label(toolbar, text=text, bg=COLOR_BG, fg=color,
                     font=("Consolas", 8, "bold")).pack(side="left", padx=5)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", padx=8, fill="y")
        self._file_label = ttk.Label(toolbar, text="(unsaved)", foreground="#888888")
        self._file_label.pack(side="left")

        # ── Paned layout ─────────────────────────────────────────────────────
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=8)
        left  = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left,  weight=1)
        paned.add(right, weight=3)

        self._build_search_panel(left)
        self._build_deck_panel(right)

        # ── Status bar ───────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready.  Load a deck or search for cards.")
        tk.Label(self, textvariable=self._status_var, bg=COLOR_PANEL, fg=COLOR_TEXT,
                 anchor="w", font=("Segoe UI", 9), padx=8, pady=3).pack(fill="x", side="bottom")

    def _build_search_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Card Search", font=("Segoe UI", 11, "bold"),
                  foreground=COLOR_HIGHLIGHT).pack(anchor="w", pady=(0, 4))

        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 4))

        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._do_search())
        entry = ttk.Entry(row, textvariable=self._search_var, font=("Segoe UI", 11))
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _: self._do_search())
        entry.focus()
        ttk.Button(row, text="✕", width=3,
                   command=lambda: self._search_var.set("")).pack(side="left", padx=(4, 0))

        # Results listbox
        lb_wrap = tk.Frame(parent, bg=COLOR_PANEL)
        lb_wrap.pack(fill="both", expand=True, pady=(0, 6))
        sb = ttk.Scrollbar(lb_wrap)
        sb.pack(side="right", fill="y")
        self._results_lb = tk.Listbox(
            lb_wrap, bg=COLOR_PANEL, fg=COLOR_TEXT,
            selectbackground=COLOR_HIGHLIGHT, selectforeground="white",
            font=("Segoe UI", 10), activestyle="none",
            yscrollcommand=sb.set, bd=0, highlightthickness=0,
        )
        self._results_lb.pack(fill="both", expand=True)
        sb.config(command=self._results_lb.yview)
        self._results_lb.bind("<Double-Button-1>", lambda _: self._add_to("main"))
        self._results_lb.bind("<Return>",          lambda _: self._add_to("main"))
        self._results_lb.bind("<Motion>",          self._on_lb_motion)
        self._results_lb.bind("<Leave>",           self._on_lb_leave)

        # Add buttons (no Side — Master Duel has no side deck)
        btn_row = ttk.Frame(parent)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="➕ Main / Auto",
                   command=lambda: self._add_to("main")).pack(
                   side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(btn_row, text="➕ Extra",
                   command=lambda: self._add_to("extra")).pack(
                   side="left", expand=True, fill="x", padx=(2, 0))

        self._result_count_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self._result_count_var,
                  foreground="#888888", font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))
        ttk.Label(parent,
                  text="Double-click or ➕ to add  •  L-click deck card to remove",
                  foreground="#555555", font=("Segoe UI", 8)).pack(anchor="w")

        self._results_data: list[int] = []
        self._populate_results(self._all_cards_by_name[:500])

    def _build_deck_panel(self, parent: ttk.Frame) -> None:
        self._section_frames: dict[str, dict] = {}
        self._section_csl: dict[str, CardScrollList] = {}

        # Master Duel has no side deck
        sections = [
            ("main",  "MAIN DECK",  MAIN_MIN, MAIN_MAX,  COLOR_OK),
            ("extra", "EXTRA DECK", 0,        EXTRA_MAX, "#3498db"),
        ]

        for sec_id, label, min_sz, max_sz, accent in sections:
            frame = ttk.Frame(parent)
            frame.pack(fill="both", expand=True, pady=(0, 6))

            # Header bar
            header = tk.Frame(frame, bg=COLOR_ACCENT)
            header.pack(fill="x")

            count_var = tk.StringVar(value=f"{label}  (0/{max_sz})")
            tk.Label(header, textvariable=count_var, bg=COLOR_ACCENT, fg="white",
                     font=("Segoe UI", 10, "bold"), padx=8, pady=4).pack(side="left")

            status_dot = tk.Label(header, text="●", bg=COLOR_ACCENT, fg=accent,
                                  font=("Segoe UI", 14), padx=4)
            status_dot.pack(side="left")

            # Scrollable card list
            def make_rm_one(s: str = sec_id) -> "callable[[int], None]":
                return lambda code: self._remove_one(s, code)

            def make_rm_all(s: str = sec_id) -> "callable[[int], None]":
                return lambda code: self._remove_all(s, code)

            csl = CardScrollList(
                frame, self._card_db, self._card_info,
                on_remove_one=make_rm_one(),
                on_remove_all=make_rm_all(),
                bg=COLOR_PANEL,
            )
            csl.pack(fill="both", expand=True)

            self._section_frames[sec_id] = {
                "count_var":  count_var,
                "status_dot": status_dot,
                "label":      label,
                "min":        min_sz,
                "max":        max_sz,
                "accent":     accent,
            }
            self._section_csl[sec_id] = csl

    # ── Search ────────────────────────────────────────────────────────────────

    def _do_search(self) -> None:
        query = self._search_var.get().strip().lower()
        if not query:
            self._populate_results(self._all_cards_by_name[:500])
            return
        matches = [(n, c) for n, c in self._all_cards_by_name if query in n.lower()]
        self._populate_results(matches[:300])

    def _populate_results(self, cards: list[tuple[str, int]]) -> None:
        lb = self._results_lb
        lb.delete(0, "end")
        self._results_data = []
        for name, code in cards:
            prefix, color = card_display_info(self._card_info, code)
            lb.insert("end", f"{prefix + ' ' if prefix else '   '}{name}")
            lb.itemconfig(lb.size() - 1, foreground=color)
            self._results_data.append(code)
        total = len(self._card_db)
        self._result_count_var.set(
            f"Showing {lb.size()} of {total} cards" if total else "No card database loaded"
        )

    # ── Listbox hover image tooltip ───────────────────────────────────────────

    def _on_lb_motion(self, event: tk.Event) -> None:
        idx = self._results_lb.nearest(event.y)
        if 0 <= idx < len(self._results_data):
            code = self._results_data[idx]
            if code != self._lb_tip_code:
                self._lb_tip_code = code
                self._cancel_lb_tip_timer()
                self._lb_tip_after = self.after(
                    400, lambda c=code, ey=event.y: self._show_lb_tip(c, ey)
                )
        else:
            self._hide_lb_tip()

    def _on_lb_leave(self, _: tk.Event) -> None:
        self._hide_lb_tip()

    def _cancel_lb_tip_timer(self) -> None:
        if self._lb_tip_after:
            self.after_cancel(self._lb_tip_after)
            self._lb_tip_after = None

    def _show_lb_tip(self, code: int, list_y: int) -> None:
        self._lb_tip_after = None
        self._hide_lb_tip()
        lb = self._results_lb
        x = lb.winfo_rootx() + lb.winfo_width() + 6
        y = lb.winfo_rooty() + list_y
        self._lb_tip = tk.Toplevel(self)
        self._lb_tip.wm_overrideredirect(True)
        self._lb_tip.wm_geometry(f"+{x}+{y}")
        self._lb_tip.configure(bg="#000000")

        if code in ImageTooltip._cache:
            self._render_lb_tip(code)
        else:
            tk.Label(self._lb_tip, text="Loading…", bg="#1a1a2e", fg="#888888",
                     padx=20, pady=16, font=("Segoe UI", 9)).pack()
            if code not in ImageTooltip._pending:
                ImageTooltip._pending.add(code)
                threading.Thread(
                    target=_fetch_card_image,
                    args=(code, lambda c=code: self._on_lb_img_ready(c)),
                    daemon=True,
                ).start()

    def _on_lb_img_ready(self, code: int) -> None:
        if self._lb_tip and self._lb_tip.winfo_exists() and self._lb_tip_code == code:
            try:
                self.after(0, lambda: self._render_lb_tip(code))
            except tk.TclError:
                pass

    def _render_lb_tip(self, code: int) -> None:
        if not (self._lb_tip and self._lb_tip.winfo_exists()):
            return
        for w in self._lb_tip.winfo_children():
            w.destroy()
        img = ImageTooltip._cache.get(code)
        if img:
            lbl = tk.Label(self._lb_tip, image=img, bg="#000000", bd=1, relief="solid")
            lbl.image = img
            lbl.pack()
        else:
            msg = (
                "No image available"
                if PIL_AVAILABLE
                else "Install Pillow for card images:\npip install Pillow"
            )
            tk.Label(self._lb_tip, text=msg, bg="#1a1a2e", fg="#888888",
                     padx=20, pady=16, font=("Segoe UI", 9)).pack()

    def _hide_lb_tip(self) -> None:
        self._cancel_lb_tip_timer()
        self._lb_tip_code = None
        if self._lb_tip:
            try:
                self._lb_tip.destroy()
            except tk.TclError:
                pass
            self._lb_tip = None

    # ── Add / remove ──────────────────────────────────────────────────────────

    def _get_selected_code(self) -> int | None:
        sel = self._results_lb.curselection()
        if not sel:
            self._set_status("Select a card from the search results first.", error=True)
            return None
        idx = sel[0]
        return self._results_data[idx] if idx < len(self._results_data) else None

    def _add_to(self, section: str) -> None:
        code = self._get_selected_code()
        if code is None:
            return
        name = self._card_db.get(code, str(code))
        # Auto-route: Extra Deck cards always go to extra regardless of which button was pressed
        if is_extra_deck_card(self._card_info, code):
            if section == "main":
                section = "extra"
                self._set_status(f"'{name}' is an Extra Deck card — routing to Extra Deck.")
        else:
            if section == "extra":
                # User explicitly clicked Extra for a non-extra card — allow it with a note
                # (edge case: keeps manual override possible)
                pass
        deck: list[int] = getattr(self, f"_{section}")
        sf = self._section_frames[section]
        copies = deck.count(code)
        if copies >= MAX_COPIES:
            self._set_status(f"'{name}' is already at {MAX_COPIES} copies.", error=True)
            return
        if len(deck) >= sf["max"]:
            self._set_status(f"{sf['label']} is full ({sf['max']} cards max).", error=True)
            return
        deck.append(code)
        self._refresh_section(section)
        self._set_status(f"Added '{name}' to {section} deck. ({deck.count(code)}/3 copies)")

    def _remove_one(self, section: str, code: int) -> None:
        deck: list[int] = getattr(self, f"_{section}")
        try:
            deck.remove(code)
        except ValueError:
            return
        self._refresh_section(section)
        name = self._card_db.get(code, str(code))
        self._set_status(f"Removed one copy of '{name}' from {section} deck.")

    def _remove_all(self, section: str, code: int) -> None:
        deck: list[int] = getattr(self, f"_{section}")
        count = deck.count(code)
        if not count:
            return
        while code in deck:
            deck.remove(code)
        self._refresh_section(section)
        name = self._card_db.get(code, str(code))
        noun = "copy" if count == 1 else "copies"
        self._set_status(f"Removed all {count} {noun} of '{name}' from {section} deck.")

    # ── Refresh display ───────────────────────────────────────────────────────

    def _refresh_section(self, section: str) -> None:
        sf  = self._section_frames[section]
        csl = self._section_csl[section]
        deck: list[int] = getattr(self, f"_{section}")

        csl.populate(deck)

        n = len(deck)
        sf["count_var"].set(f"{sf['label']}  ({n}/{sf['max']})")

        if section == "main":
            if n < sf["min"]:
                dot_color = COLOR_WARN
            elif n > sf["max"]:
                dot_color = COLOR_ERROR
            else:
                dot_color = COLOR_OK
        else:
            dot_color = COLOR_ERROR if n > sf["max"] else sf["accent"]
        sf["status_dot"].config(fg=dot_color)

    def _refresh_all(self) -> None:
        for section in ("main", "extra"):
            self._refresh_section(section)

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> list[str]:
        from collections import Counter
        errors: list[str] = []
        n = len(self._main)
        if n < MAIN_MIN:
            errors.append(f"Main deck has {n} cards (minimum {MAIN_MIN})")
        if n > MAIN_MAX:
            errors.append(f"Main deck has {n} cards (maximum {MAIN_MAX})")
        if len(self._extra) > EXTRA_MAX:
            errors.append(f"Extra deck has {len(self._extra)} cards (maximum {EXTRA_MAX})")
        # No side deck in Master Duel — only count main copies
        for code, count in Counter(self._main).items():
            if count > MAX_COPIES:
                name = self._card_db.get(code, str(code))
                errors.append(f"'{name}' appears {count} times in Main (max {MAX_COPIES})")
        return errors

    # ── File operations ───────────────────────────────────────────────────────

    def _load_ydk(self) -> None:
        path = filedialog.askopenfilename(
            title="Open YDK Deck File",
            filetypes=[("YGOPro Deck", "*.ydk"), ("All files", "*.*")],
            initialdir=str(ENGINES_DIR) if ENGINES_DIR.exists() else str(ROOT),
        )
        if not path:
            return
        p = Path(path)
        try:
            main, extra, side = parse_ydk(p)
            self._main  = main
            self._extra = extra
            # side deck silently discarded (MD has no side deck)
            self._current_file = p
            self._file_label.config(text=str(p), foreground=COLOR_TEXT)
            self.title(f"YGO Deck Editor — {p.name}")
            self._refresh_all()
            msg = f"Loaded '{p.name}' — {len(self._main)} main, {len(self._extra)} extra."
            if side:
                msg += f"  ({len(side)} side-deck cards ignored — no side deck in MD)"
            self._set_status(msg)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _save_ydk(self) -> None:
        if self._current_file:
            self._do_save(self._current_file)
        else:
            self._save_ydk_as()

    def _save_ydk_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Deck As",
            defaultextension=".ydk",
            filetypes=[("YGOPro Deck", "*.ydk"), ("All files", "*.*")],
            initialdir=str(ENGINES_DIR) if ENGINES_DIR.exists() else str(ROOT),
        )
        if not path:
            return
        self._do_save(Path(path))

    def _do_save(self, path: Path) -> None:
        errors = self._validate()
        if errors:
            msg = "Deck has issues:\n\n" + "\n".join(f"• {e}" for e in errors)
            msg += "\n\nSave anyway?"
            if not messagebox.askyesno("Validation Warning", msg, icon="warning"):
                return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_ydk(path, self._main, self._extra, [])   # no side deck in MD
            self._current_file = path
            self._file_label.config(text=str(path), foreground=COLOR_TEXT)
            self.title(f"YGO Deck Editor — {path.name}")
            self._set_status(
                f"Saved '{path.name}' — {len(self._main)} main / {len(self._extra)} extra."
            )
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def _clear_deck(self) -> None:
        if self._main or self._extra:
            if not messagebox.askyesno("Clear Deck", "Clear all cards from the deck?"):
                return
        self._main.clear()
        self._extra.clear()
        self._current_file = None
        self._file_label.config(text="(unsaved)", foreground="#888888")
        self.title("YGO Deck Editor")
        self._refresh_all()
        self._set_status("Deck cleared.")

    # ── Card database refresh ─────────────────────────────────────────────────

    def _refresh_cards_async(self) -> None:
        self._refresh_btn.config(state="disabled", text="🔄 Updating…")
        self._set_status("Fetching latest card data from YGOPRODeck API…")
        threading.Thread(target=self._do_refresh, daemon=True).start()

    def _do_refresh(self) -> None:
        try:
            req = urllib.request.Request(
                YGOPRODECK_URL, headers={"User-Agent": "ygo-meta-ai/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            # ── Build dicts (ban status from OCG ≈ MD) ───────────────────────
            db:   dict[int, str]  = {}
            info: dict[str, dict] = {}
            _extra_kw = ("fusion", "synchro", "xyz", "link")

            for card in raw.get("data", []):
                cid:  int = card["id"]
                name: str = card["name"]
                db[cid] = name

                misc     = card.get("misc_info", [{}])[0] if card.get("misc_info") else {}
                in_md    = "Master Duel" in misc.get("formats", [])
                ctype    = card.get("type", "").lower()
                is_extra = any(kw in ctype for kw in _extra_kw)
                ban_raw  = card.get("banlist_info", {}).get("ban_ocg")
                ban_ocg  = "Forbidden" if ban_raw == "Banned" else ban_raw
                entry    = {
                    "in_md":    in_md,
                    "ban_ocg":  ban_ocg,
                    "is_extra": is_extra,
                }
                info[str(cid)] = entry

                for img in card.get("card_images", []):
                    alt_id = img.get("id")
                    if alt_id and alt_id != cid:
                        db[alt_id] = name
                        info[str(alt_id)] = entry

            # ── Write to disk ────────────────────────────────────────────────
            CARD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CARD_DB_PATH, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in db.items()}, f, ensure_ascii=False, indent=2)
            with open(CARD_INFO_PATH, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

            ban_count = sum(1 for v in info.values() if v["ban_ocg"])
            self.after(0, lambda: self._apply_refreshed_db(db, info, ban_count))

        except Exception as exc:
            self.after(0, lambda: self._refresh_failed(str(exc)))

    def _apply_refreshed_db(
        self, db: dict[int, str], info: dict[str, dict], ban_count: int
    ) -> None:
        prev = len(self._card_db)
        self._card_db   = db
        self._card_info = info
        self._all_cards_by_name = sorted(
            ((v, k) for k, v in db.items()), key=lambda x: x[0].lower()
        )
        # Propagate new references to all CardScrollList widgets
        for csl in self._section_csl.values():
            csl.card_info = info
            csl._card_db  = db
        self._do_search()
        self._refresh_all()
        self._refresh_btn.config(state="normal", text="🔄 Update Cards")
        added = len(db) - prev
        sign  = f"+{added}" if added >= 0 else str(added)
        self._set_status(
            f"Updated — {len(db):,} cards ({sign}).  MD restricted: {ban_count} cards."
        )

    def _refresh_failed(self, error: str) -> None:
        self._refresh_btn.config(state="normal", text="🔄 Update Cards")
        self._set_status(f"Update failed: {error}", error=True)
        messagebox.showerror("Update Failed", f"Could not fetch cards:\n\n{error}")

    # ── Status bar ────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_var.set(msg)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    app = DeckEditor()
    # Open a YDK file passed as CLI argument
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists() and p.suffix.lower() == ".ydk":
            try:
                main, extra, side = parse_ydk(p)
                app._main = main
                app._extra = extra
                app._current_file = p
                app._file_label.config(text=str(p), foreground=COLOR_TEXT)
                app.title(f"YGO Deck Editor — {p.name}")
                app._refresh_all()
                app._set_status(f"Loaded '{p.name}' — {len(main)} main, {len(extra)} extra.")
            except Exception as exc:
                messagebox.showerror("Load Error", str(exc))
    app.mainloop()
