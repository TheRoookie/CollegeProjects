# Docs — Index

Complete specification for recreating the **Car Parking Space Counter** project from scratch.

| File | Contents |
|------|---------|
| [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md) | Purpose, syllabus topics, tech stack, architecture diagram, design decisions |
| [02_FILE_STRUCTURE.md](02_FILE_STRUCTURE.md) | Full directory tree, file roles, import graph, config loading chain |
| [03_ENVIRONMENT_SETUP.md](03_ENVIRONMENT_SETUP.md) | Step-by-step setup: Python version, venv, pip install, CPU torch, verification |
| [04_CONFIG_REFERENCE.md](04_CONFIG_REFERENCE.md) | Every config.yaml key with type, default, range, and effect |
| [05_MODULE_video_handler.md](05_MODULE_video_handler.md) | `VideoHandler` class: constructor, `frames()` generator, FPS cap, resize |
| [06_MODULE_roi_manager.md](06_MODULE_roi_manager.md) | `ParkingSpot` dataclass + debounce, `ROIManager`: load/save/extract_patch/draw |
| [07_MODULE_detector.md](07_MODULE_detector.md) | `Detection` dataclass, `VehicleDetector`: lazy load, YOLO inference, COCO IDs |
| [08_MODULE_tracker.md](08_MODULE_tracker.md) | `Track` dataclass, `CentroidTracker`: Hungarian algorithm step-by-step |
| [09_MODULE_classifier.md](09_MODULE_classifier.md) | `PixelCountClassifier` (all guards), `CNNClassifier` (PyTorch), `SpaceClassifier` façade |
| [10_MODULE_pipeline.md](10_MODULE_pipeline.md) | `PipelineThread`: signals, `detect_every`, `_detector_failed`, `_detection_overlap()` |
| [11_MODULE_ui.md](11_MODULE_ui.md) | `VideoLabel`, `AnalyticsSidebar`, `MainWindow`: toolbar, pipeline wiring, classifier fix |
| [12_MODULE_draw_rois.md](12_MODULE_draw_rois.md) | Standalone ROI tool: mouse callbacks, key bindings, process_width sync, JSON output |
| [13_BUGS_AND_FIXES.md](13_BUGS_AND_FIXES.md) | 7 bugs with root cause analysis and exact code fixes |
| [14_DATA_FORMATS.md](14_DATA_FORMATS.md) | `parking_spots.json` schema, full annotated `config.yaml`, internal data structures |

## Documentation Maintenance Rule

> **Any change made to the project — code, config, data formats, bug fixes, or module behaviour — MUST be reflected in the corresponding document(s) in this `/Docs` folder before the change is considered complete.**
>
> - New feature or module → create or update the relevant `0X_MODULE_*.md` file
> - Bug discovered/fixed → add an entry to `13_BUGS_AND_FIXES.md`
> - Config key added/changed → update `04_CONFIG_REFERENCE.md`
> - Data format change → update `14_DATA_FORMATS.md`
> - File structure change → update `02_FILE_STRUCTURE.md`
> - Environment/dependency change → update `03_ENVIRONMENT_SETUP.md`
> - New doc added → register it in the table above in `00_INDEX.md`

---

## Reading Order for Full Reproduction

1. Start with **01** (overview) and **02** (file structure) to understand the project
2. Follow **03** (environment) to set up your Python environment
3. Create all source files using **05–12** (one doc per module)
4. Use **04** (config) and **14** (data formats) as references while coding
5. Refer to **13** (bugs) if you encounter any of the documented issues
