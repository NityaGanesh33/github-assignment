# Differential equations

Notebooks for CHME 5137, Lecture 8 (2026): solving ODEs with SciPy's `solve_ivp`.

| Notebook | What it is |
|---|---|
| `solve-ivp-intro.ipynb` | Worked examples, done in class: decay, `rtol` and `atol`, a packed bed, a stiff reaction |
| `rabbits-and-foxes.ipynb` | The exercise. Started in class, finished for homework |

Each notebook has a paired `.py` file holding the same cells as plain text (see
**Jupytext** below).

## Getting started

1. **Fork** this repository to your own GitHub account (button at top right).
2. **Clone your fork**, not this one:
   ```bash
   git clone git@github.com:YOUR-USERNAME/differential-equations.git   # SSH, e.g. on Explorer
   # or
   git clone https://github.com/YOUR-USERNAME/differential-equations.git
   ```
   On Explorer, clone it into `/courses/CHME5137.202710/students/$USER/`.
3. Open `rabbits-and-foxes.ipynb` and work through it, using your own kernel
   (`chme5137` on Explorer). Commit and push as you go:
   ```bash
   git add rabbits-and-foxes.ipynb rabbits-and-foxes.py
   git commit -m "Task 1: solved with default settings and plotted"
   git push
   ```
   This is your own fork, so committing on `main` is fine. Don't open a pull request
   back to `CHME5137/differential-equations`: that would publish your answers to everyone.

## Jupytext

`jupytext.toml` pairs each notebook with a `.py` script in the *percent* format. When
jupytext is installed, saving the notebook updates the `.py` as well. The `.py` diffs and
merges cleanly, and you can run it with `python rabbits-and-foxes.py`. Lecture 9 covers
this properly. Read
[notebooks-and-git.md](https://github.com/CHME5137/github-assignment/blob/main/notebooks-and-git.md)
before then.

- **If jupytext isn't installed yet**, just edit the `.ipynb`. The `.py` gets out of
  date, which is fine for now. Once jupytext is installed, run
  `jupytext --sync rabbits-and-foxes.ipynb` once to bring the `.py` back in line.
- **In VS Code**, you can open the `.py` directly and run its `# %%` cells one at a
  time. The Jupytext Sync extension keeps the two files in step.

---

*The 2026 revision was written with Claude Code (Claude Opus 5.5) and checked by
Prof. West.*
