# BEEF GUI

A browser-based graphical user interface for [beef](https://github.com/knutankv/beef) (BEam Elements Framework).

The GUI runs a small local Python server that holds a live Python session. The page in your browser drives that
session: every analysis runs the **installed** `beef` package in a background job, with its output shown live in
the page (and in the terminal). Update beef and the GUI uses the new version — nothing is bundled.

## Start

The GUI is part of the `beef` package (`beef.gui`):

```bash
pip install -e .            # from the repository root (or pip install beefpy)

beef-gui                    # or: python -m beef.gui
```

From Python: `import beef.gui; beef.gui.launch()`.

A browser tab opens at `http://localhost:8070/?token=…`. Stop the server with Ctrl+C in the terminal.
Options: `--port 8080`, `--no-browser`, `--host 0.0.0.0` (expose on the network — see *Security*),
`--token <fixed token>`, `--verbose`. Only the standard library is used for the server.

## What you can do

| Area | |
|---|---|
| **Model** | 2D/3D nodes and elements (tables, line generator, chain nodes), parts → `fe.Part` / `fe.Assembly`; *Refine mesh* (all, selected or listed elements) with `ElDef.refine_mesh`, one-step undo |
| **Sections** | `fe.Section` with E, ρ, A, I_y, I_z, J, ν, m, G, κ, shear deformation, mass formulation, orientation `e2` |
| **Supports** | `fe.Constraint` to ground or node-to-node ties |
| **Features** | `fe.Spring`, `fe.Dashpot`, `fe.PointMass`, full coupled matrices `fe.CustomMatrix` |
| **Loads** | `fe.Force`: constant, ramp, or time history `a(t)`; element loads (self weight, mass per length, total mass or an expression in `sec`) on all elements, a list/selection or a section, via `fe.create_nodal_load_from_element_load` |
| **Analysis** | linear/nonlinear static, modal (`run_eig`), linear/nonlinear dynamic (Newmark), linear buckling |
| **Results** | deformed shape / mode shapes with animation, section forces N, Vy, Vz, T, My, Mz (2D: N, V, M) as diagrams in each element's local plane and as min/max tables, reactions, time histories, node/element inspector |
| **Python** | export the model + analysis as a plain beef script; a Python console in the same session (`eldef`, `analysis`, `result`, `np`, `fe`, `beef`) |

Node fields in Supports, Features and Loads accept typing (`1,4,7-9`) or picking: click into the field, then click
nodes in the view to add or remove them.

**Static base state** (Analysis tab, for modal and linear dynamic analysis): loads get a *case* — *analysis* or
*static base (preload)*. With *Base = linear static*, the base loads are solved linearly and the following analysis
uses the undeformed geometry with the geometric stiffness K_g(N₀) of the base axial forces (`el.N0`,
`include_linear_kg`). With *Base = nonlinear static*, `run_static` is run in load steps and the following analysis
uses the deformed eldef, whose element stiffness is the tangent stiffness including the geometric stiffness. Linear
dynamic results are reported as base + increment. Nonlinear dynamic analysis cannot follow a base state, because
beef's `run_dynamic` always starts from the undeformed state at rest. The exported script contains the same base step.

**Selection and visibility** (Model tab and *Isolate* in the view bar): select elements by clicking (shift/ctrl-click
to add/remove), by ctrl-dragging a box, by typing a list, or all elements with a given section. Then show only the
selection, hide it, invert it, or limit the element table to it. Diagrams and fit follow the visible elements.

Colours: structure green, loads blue (arrows point in the signed direction, length scales with magnitude, labelled
with component and value), supports white, deformed shape orange, selection white-cyan.

**Solver.** *beef* uses `fe.Analysis` directly. *sparse* uses beef's element and feature matrices but assembles them
sparsely and imposes constraints by DOF elimination (modal: undamped modes via shift-invert Lanczos; linear static).
It gives the same results to round-off and is much faster for large models. *auto* (default) chooses sparse for
modal and linear static analysis above 1500 DOFs.

## Model files and examples

`.beef` files hold a GUI model as JSON (*Open* / *Save*). The *Examples* list shows every `.beef` file in
`beef/gui/examples/` and in the folders listed in the `BEEF_GUI_EXAMPLES` environment variable (separated by
`;` on Windows, `:` elsewhere) — add your own models there.

Included: small 2D/3D teaching models and the Bergsøysund floating bridge (truss part of the Abaqus model used
in the WAWI examples, 1210 nodes; see the `notes` field in the file for how it was derived).

## Security

The server executes Python (it is a front end to a Python session, with a console). By default it listens on
127.0.0.1 only, every API request must carry the random session token from the start-up URL, and requests with a
foreign `Host` header are refused. Only use `--host 0.0.0.0` on a trusted network.

## Layout

```
beef/gui/
  server.py     HTTP server, token check, routing
  session.py    namespace, background job queue, stdout capture, cancellation
  bootstrap.py  imports beef, builds the session namespace
  api.py        endpoints: info, examples, run, script, element results, console, jobs
  model.py      GUI model (dict) -> beef code (used both to build the model and for export)
  solve.py      runs analyses, section forces, reactions
  sparse.py     sparse assembly and solvers for large models
  examples.py   lists/loads the .beef example files
  examples/     *.beef example models
  static/       index.html, app.css, app.js
```
