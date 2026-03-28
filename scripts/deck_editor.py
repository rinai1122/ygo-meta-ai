"""
YGO Deck Editor — GUI for creating and editing YDK deck files.

Usage:
    python scripts/deck_editor.py

No extra dependencies beyond Python's built-in tkinter.
Card names are loaded from data/card_names.json (run build_card_db.py first).
"""

from __future__ import annotations

import json
import threading
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

YGOPRODECK_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php?misc=yes"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
CARD_DB_PATH = ROOT / "data" / "card_names.json"
ENGINES_DIR = ROOT / "data" / "engines"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAIN_MIN, MAIN_MAX = 40, 60
EXTRA_MAX = 15
SIDE_MAX = 15
MAX_COPIES = 3

COLOR_OK = "#2ecc71"
COLOR_WARN = "#e67e22"
COLOR_ERROR = "#e74c3c"
COLOR_BG = "#1a1a2e"
COLOR_PANEL = "#16213e"
COLOR_ACCENT = "#0f3460"
COLOR_TEXT = "#e0e0e0"
COLOR_HIGHLIGHT = "#e94560"
COLOR_ENTRY = "#0f3460"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_card_db() -> dict[int, str]:
    if not CARD_DB_PATH.exists():
        return {}
    with open(CARD_DB_PATH, encoding="utf-8") as f:
        raw: dict[str, str] = json.load(f)
    return {int(k): v for k, v in raw.items()}


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


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class DeckEditor(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("YGO Deck Editor")
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.configure(bg=COLOR_BG)

        self._card_db: dict[int, str] = load_card_db()
        # Reverse lookup: lowercase name -> code (for search)
        self._name_to_code: dict[str, int] = {
            v.lower(): k for k, v in self._card_db.items()
        }
        # Sorted list of all (name, code) for display
        self._all_cards: list[tuple[str, int]] = sorted(
            self._card_db.items(), key=lambda x: x[1]
        )
        # name sorted for better search
        self._all_cards_by_name: list[tuple[str, int]] = sorted(
            ((v, k) for k, v in self._card_db.items()), key=lambda x: x[0].lower()
        )

        # Deck state: lists of card codes (with repetition for copies)
        self._main: list[int] = []
        self._extra: list[int] = []
        self._side: list[int] = []

        self._current_file: Path | None = None
        self._build_ui()
        self._refresh_all()

        if not self._card_db:
            messagebox.showwarning(
                "Card database missing",
                "data/card_names.json not found.\n"
                "Run: python scripts/build_card_db.py\n\n"
                "You can still load/edit YDK files, but card names won't show.",
            )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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
        style.configure("TNotebook", background=COLOR_BG, tabmargins=[2, 5, 2, 0])
        style.configure("TNotebook.Tab", background=COLOR_ACCENT, foreground=COLOR_TEXT, padding=[10, 4])
        style.map("TNotebook.Tab", background=[("selected", COLOR_HIGHLIGHT)])
        style.configure("TScrollbar", background=COLOR_ACCENT, troughcolor=COLOR_PANEL,
                        arrowcolor=COLOR_TEXT)

        # --- Top toolbar ---
        toolbar = ttk.Frame(self, style="TFrame")
        toolbar.pack(fill="x", padx=8, pady=(8, 0))

        ttk.Button(toolbar, text="📂 Load YDK", command=self._load_ydk).pack(side="left", padx=2)
        ttk.Button(toolbar, text="💾 Save YDK", command=self._save_ydk, style="Accent.TButton").pack(side="left", padx=2)
        ttk.Button(toolbar, text="💾 Save As", command=self._save_ydk_as).pack(side="left", padx=2)
        ttk.Button(toolbar, text="🗑 Clear Deck", command=self._clear_deck).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", padx=8, fill="y")
        self._refresh_btn = ttk.Button(toolbar, text="🔄 Update Cards", command=self._refresh_cards_async)
        self._refresh_btn.pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", padx=8, fill="y")

        self._file_label = ttk.Label(toolbar, text="(unsaved)", foreground="#888888")
        self._file_label.pack(side="left")

        # --- Main layout: left search panel + right deck panel ---
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        self._build_search_panel(left)
        self._build_deck_panel(right)

        # --- Status bar ---
        self._status_var = tk.StringVar(value="Ready. Load or build a deck.")
        status_bar = tk.Label(self, textvariable=self._status_var,
                              bg=COLOR_PANEL, fg=COLOR_TEXT, anchor="w",
                              font=("Segoe UI", 9), padx=8, pady=3)
        status_bar.pack(fill="x", side="bottom")

    def _build_search_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Card Search", font=("Segoe UI", 11, "bold"),
                  foreground=COLOR_HIGHLIGHT).pack(anchor="w", pady=(0, 4))

        search_row = ttk.Frame(parent)
        search_row.pack(fill="x", pady=(0, 4))

        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._do_search())
        search_entry = ttk.Entry(search_row, textvariable=self._search_var, font=("Segoe UI", 11))
        search_entry.pack(side="left", fill="x", expand=True)
        search_entry.bind("<Return>", lambda _: self._do_search())
        search_entry.focus()

        ttk.Button(search_row, text="✕", width=3,
                   command=lambda: self._search_var.set("")).pack(side="left", padx=(4, 0))

        # Results listbox
        results_frame = tk.Frame(parent, bg=COLOR_PANEL, bd=1, relief="flat")
        results_frame.pack(fill="both", expand=True, pady=(0, 6))

        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side="right", fill="y")

        self._results_lb = tk.Listbox(
            results_frame, bg=COLOR_PANEL, fg=COLOR_TEXT,
            selectbackground=COLOR_HIGHLIGHT, selectforeground="white",
            font=("Segoe UI", 10), activestyle="none",
            yscrollcommand=scrollbar.set, bd=0, highlightthickness=0,
        )
        self._results_lb.pack(fill="both", expand=True)
        scrollbar.config(command=self._results_lb.yview)
        self._results_lb.bind("<Double-Button-1>", lambda _: self._add_to("main"))
        self._results_lb.bind("<Return>", lambda _: self._add_to("main"))

        # Add buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x")

        ttk.Button(btn_frame, text="➕ Main", command=lambda: self._add_to("main")).pack(
            side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(btn_frame, text="➕ Extra", command=lambda: self._add_to("extra")).pack(
            side="left", expand=True, fill="x", padx=2)
        ttk.Button(btn_frame, text="➕ Side", command=lambda: self._add_to("side")).pack(
            side="left", expand=True, fill="x", padx=(2, 0))

        # Search result count
        self._result_count_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self._result_count_var,
                  foreground="#888888", font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))

        # Hint
        ttk.Label(parent, text="Double-click or use ➕ buttons to add cards",
                  foreground="#666666", font=("Segoe UI", 8)).pack(anchor="w")

        # Load initial results
        self._populate_results(self._all_cards_by_name[:500])

    def _build_deck_panel(self, parent: ttk.Frame) -> None:
        # Three sections stacked vertically with labels and listboxes
        self._section_frames: dict[str, dict] = {}

        sections = [
            ("main",  "MAIN DECK",  MAIN_MIN, MAIN_MAX,  COLOR_OK),
            ("extra", "EXTRA DECK", 0,        EXTRA_MAX, "#3498db"),
            ("side",  "SIDE DECK",  0,        SIDE_MAX,  "#9b59b6"),
        ]

        for sec_id, label, min_sz, max_sz, accent in sections:
            frame = ttk.Frame(parent)
            frame.pack(fill="both", expand=True, pady=(0, 6))

            header = tk.Frame(frame, bg=COLOR_ACCENT)
            header.pack(fill="x")

            count_var = tk.StringVar(value=f"{label}  (0/{max_sz})")
            tk.Label(header, textvariable=count_var,
                     bg=COLOR_ACCENT, fg="white",
                     font=("Segoe UI", 10, "bold"), padx=8, pady=4).pack(side="left")

            status_dot = tk.Label(header, text="●", bg=COLOR_ACCENT, fg=accent,
                                  font=("Segoe UI", 14), padx=4)
            status_dot.pack(side="left")

            btn_row = tk.Frame(header, bg=COLOR_ACCENT)
            btn_row.pack(side="right", padx=4)

            def make_remove(s=sec_id):
                return lambda: self._remove_selected(s)

            def make_copy(s=sec_id):
                return lambda: self._copy_to_clipboard(s)

            tk.Button(btn_row, text="−", command=make_remove(),
                      bg=COLOR_HIGHLIGHT, fg="white", relief="flat",
                      font=("Segoe UI", 10, "bold"), padx=6, pady=1,
                      cursor="hand2").pack(side="left", padx=2, pady=2)

            lb_frame = tk.Frame(frame, bg=COLOR_PANEL)
            lb_frame.pack(fill="both", expand=True)

            sb = ttk.Scrollbar(lb_frame)
            sb.pack(side="right", fill="y")

            lb = tk.Listbox(
                lb_frame, bg=COLOR_PANEL, fg=COLOR_TEXT,
                selectbackground=COLOR_HIGHLIGHT, selectforeground="white",
                font=("Segoe UI", 10), activestyle="none",
                yscrollcommand=sb.set, bd=0, highlightthickness=0,
            )
            lb.pack(fill="both", expand=True)
            sb.config(command=lb.yview)
            lb.bind("<Delete>", lambda _, s=sec_id: self._remove_selected(s))
            lb.bind("<BackSpace>", lambda _, s=sec_id: self._remove_selected(s))

            self._section_frames[sec_id] = {
                "lb": lb,
                "count_var": count_var,
                "status_dot": status_dot,
                "label": label,
                "min": min_sz,
                "max": max_sz,
                "accent": accent,
            }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _do_search(self) -> None:
        query = self._search_var.get().strip().lower()
        if not query:
            self._populate_results(self._all_cards_by_name[:500])
            return
        matches = [
            (name, code) for name, code in self._all_cards_by_name
            if query in name.lower()
        ]
        self._populate_results(matches[:300])

    def _populate_results(self, cards: list[tuple[str, int]]) -> None:
        lb = self._results_lb
        lb.delete(0, "end")
        self._results_data: list[int] = []
        for name, code in cards:
            lb.insert("end", f"  {name}")
            self._results_data.append(code)
        count = lb.size()
        total = len(self._card_db)
        self._result_count_var.set(
            f"Showing {count} of {total} cards" if total else "No card database loaded"
        )

    # ------------------------------------------------------------------
    # Add / remove
    # ------------------------------------------------------------------

    def _get_selected_code(self) -> int | None:
        sel = self._results_lb.curselection()
        if not sel:
            self._set_status("Select a card from the search results first.", error=True)
            return None
        idx = sel[0]
        if idx >= len(self._results_data):
            return None
        return self._results_data[idx]

    def _add_to(self, section: str) -> None:
        code = self._get_selected_code()
        if code is None:
            return
        deck = getattr(self, f"_{section}")
        copies = deck.count(code)
        if copies >= MAX_COPIES:
            name = self._card_db.get(code, str(code))
            self._set_status(f"Can't add '{name}' — already at {MAX_COPIES} copies.", error=True)
            return
        max_sz = self._section_frames[section]["max"]
        if len(deck) >= max_sz:
            label = self._section_frames[section]["label"]
            self._set_status(f"{label} is full ({max_sz} cards max).", error=True)
            return
        deck.append(code)
        self._refresh_section(section)
        name = self._card_db.get(code, str(code))
        self._set_status(f"Added '{name}' to {section} deck. ({deck.count(code)}/3 copies)")

    def _remove_selected(self, section: str) -> None:
        sf = self._section_frames[section]
        lb: tk.Listbox = sf["lb"]
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        deck: list[int] = getattr(self, f"_{section}")
        if idx < len(deck):
            code = deck[idx]
            deck.pop(idx)
            self._refresh_section(section)
            name = self._card_db.get(code, str(code))
            self._set_status(f"Removed one copy of '{name}' from {section} deck.")

    # ------------------------------------------------------------------
    # Refresh display
    # ------------------------------------------------------------------

    def _refresh_section(self, section: str) -> None:
        sf = self._section_frames[section]
        lb: tk.Listbox = sf["lb"]
        deck: list[int] = getattr(self, f"_{section}")

        lb.delete(0, "end")
        # Group by card code preserving order of first occurrence
        seen: dict[int, int] = {}
        order: list[int] = []
        for code in deck:
            if code not in seen:
                seen[code] = 0
                order.append(code)
            seen[code] += 1

        for code in order:
            name = self._card_db.get(code, f"#{code}")
            copies = seen[code]
            bullets = "●" * copies + "○" * (MAX_COPIES - copies)
            lb.insert("end", f"  {bullets}  {name}")

        # Update header
        n = len(deck)
        min_sz = sf["min"]
        max_sz = sf["max"]
        label = sf["label"]
        sf["count_var"].set(f"{label}  ({n}/{max_sz})")

        # Status dot color
        if section == "main":
            if n < min_sz:
                color = COLOR_WARN
            elif n > max_sz:
                color = COLOR_ERROR
            else:
                color = COLOR_OK
        else:
            color = COLOR_ERROR if n > max_sz else sf["accent"]
        sf["status_dot"].config(fg=color)

    def _refresh_all(self) -> None:
        for section in ("main", "extra", "side"):
            self._refresh_section(section)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> list[str]:
        errors = []
        n = len(self._main)
        if n < MAIN_MIN:
            errors.append(f"Main deck has {n} cards (minimum {MAIN_MIN})")
        if n > MAIN_MAX:
            errors.append(f"Main deck has {n} cards (maximum {MAIN_MAX})")
        if len(self._extra) > EXTRA_MAX:
            errors.append(f"Extra deck has {len(self._extra)} cards (maximum {EXTRA_MAX})")
        if len(self._side) > SIDE_MAX:
            errors.append(f"Side deck has {len(self._side)} cards (maximum {SIDE_MAX})")
        from collections import Counter
        combined = Counter(self._main + self._side)
        for code, count in combined.items():
            if count > MAX_COPIES:
                name = self._card_db.get(code, str(code))
                errors.append(f"'{name}' appears {count} times (max {MAX_COPIES})")
        return errors

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

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
            self._main, self._extra, self._side = parse_ydk(p)
            self._current_file = p
            self._file_label.config(text=str(p), foreground=COLOR_TEXT)
            self.title(f"YGO Deck Editor — {p.name}")
            self._refresh_all()
            self._set_status(f"Loaded '{p.name}' — {len(self._main)} main, {len(self._extra)} extra, {len(self._side)} side.")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

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
            write_ydk(path, self._main, self._extra, self._side)
            self._current_file = path
            self._file_label.config(text=str(path), foreground=COLOR_TEXT)
            self.title(f"YGO Deck Editor — {path.name}")
            self._set_status(f"Saved to '{path.name}' — {len(self._main)} main / {len(self._extra)} extra / {len(self._side)} side.")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _clear_deck(self) -> None:
        if (self._main or self._extra or self._side):
            if not messagebox.askyesno("Clear Deck", "Clear all cards from the deck?"):
                return
        self._main.clear()
        self._extra.clear()
        self._side.clear()
        self._current_file = None
        self._file_label.config(text="(unsaved)", foreground="#888888")
        self.title("YGO Deck Editor")
        self._refresh_all()
        self._set_status("Deck cleared.")

    def _copy_to_clipboard(self, section: str) -> None:
        pass  # placeholder

    # ------------------------------------------------------------------
    # Card database refresh (fetch from YGOPRODeck in background thread)
    # ------------------------------------------------------------------

    def _refresh_cards_async(self) -> None:
        self._refresh_btn.config(state="disabled", text="🔄 Updating…")
        self._set_status("Fetching latest card list from YGOPRODeck API…")
        threading.Thread(target=self._do_refresh, daemon=True).start()

    def _do_refresh(self) -> None:
        try:
            req = urllib.request.Request(
                YGOPRODECK_URL,
                headers={"User-Agent": "ygo-meta-ai/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            db: dict[int, str] = {}
            for card in raw.get("data", []):
                db[card["id"]] = card["name"]
                for img in card.get("card_images", []):
                    alt_id = img.get("id")
                    if alt_id and alt_id != card["id"]:
                        db[alt_id] = card["name"]

            # Write to disk
            CARD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CARD_DB_PATH, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in db.items()}, f, ensure_ascii=False, indent=2)

            # Schedule UI update back on main thread
            self.after(0, lambda: self._apply_refreshed_db(db))

        except Exception as exc:
            self.after(0, lambda: self._refresh_failed(str(exc)))

    def _apply_refreshed_db(self, db: dict[int, str]) -> None:
        prev = len(self._card_db)
        self._card_db = db
        self._name_to_code = {v.lower(): k for k, v in db.items()}
        self._all_cards_by_name = sorted(
            ((v, k) for k, v in db.items()), key=lambda x: x[0].lower()
        )
        # Re-run current search so results reflect new cards
        self._do_search()
        self._refresh_btn.config(state="normal", text="🔄 Update Cards")
        added = len(db) - prev
        sign = f"+{added}" if added >= 0 else str(added)
        self._set_status(
            f"Card database updated — {len(db):,} cards total ({sign} since last update). "
            f"Saved to data/card_names.json."
        )

    def _refresh_failed(self, error: str) -> None:
        self._refresh_btn.config(state="normal", text="🔄 Update Cards")
        self._set_status(f"Update failed: {error}", error=True)
        messagebox.showerror("Update Failed", f"Could not fetch cards from YGOPRODeck:\n\n{error}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_var.set(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = DeckEditor()
    app.mainloop()
