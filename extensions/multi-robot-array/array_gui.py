from __future__ import annotations

import sys
from pathlib import Path
import importlib.util

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QProcess


def _load_generate_array_module() -> object:
    module_path = Path(__file__).with_name("generate_array.py")
    spec = importlib.util.spec_from_file_location("generate_array", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load generate_array.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



DEFAULT_OFFSET_Y_MM = 80.0
DEFAULT_COUNT = 6
DEFAULT_BASE_ROT_DEG = 0.0
DEFAULT_TILT_X_DEG = -30.0


class ArrayGui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Robot Array Preview")
        self.resize(380, 540)

        self._generate = _load_generate_array_module()
        self._preview_xml_path: Path | None = None
        self._viewer_processes: list[QProcess] = []

        root = QWidget()
        root_layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        self.right_panel = self._build_right_panel()
        root_layout.addWidget(self.right_panel, 1)

        self._load_default_xml()

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("Parameters")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        xml_group = QGroupBox("XML File")
        xml_layout = QHBoxLayout(xml_group)
        self.xml_path = QLineEdit()
        self.xml_path.setPlaceholderText("Select MuJoCo XML")
        self.xml_path.editingFinished.connect(self._xml_path_changed)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._pick_xml)
        xml_layout.addWidget(self.xml_path, 1)
        xml_layout.addWidget(browse_btn)
        layout.addWidget(xml_group)

        offset_group = QGroupBox("Offset Y (mm)")
        offset_layout = QFormLayout(offset_group)
        self.offset_y = self._spinbox(-10000, 10000, DEFAULT_OFFSET_Y_MM)
        offset_layout.addRow("Y", self.offset_y)
        layout.addWidget(offset_group)

        array_rot_group = QGroupBox("Array Rotation / Tilt (deg)")
        array_rot_layout = QFormLayout(array_rot_group)
        self.base_rot = self._spinbox(-360, 360, DEFAULT_BASE_ROT_DEG)
        self.tilt_x = self._spinbox(-360, 360, DEFAULT_TILT_X_DEG)
        array_rot_layout.addRow("Base Rot Z", self.base_rot)
        array_rot_layout.addRow("Tilt X", self.tilt_x)
        layout.addWidget(array_rot_group)

        array_group = QGroupBox("Array")
        array_layout = QFormLayout(array_group)
        self.count = QSpinBox()
        self.count.setRange(1, 128)
        self.count.setValue(DEFAULT_COUNT)
        array_layout.addRow("Count", self.count)
        layout.addWidget(array_group)

        self.view_input_btn = QPushButton("Open MuJoCo Viewer (Input)")
        self.view_input_btn.clicked.connect(self._open_input_viewer)
        layout.addWidget(self.view_input_btn)

        self.view_array_btn = QPushButton("Open MuJoCo Viewer (Array)")
        self.view_array_btn.clicked.connect(self._open_array_viewer)
        layout.addWidget(self.view_array_btn)

        self.generate_btn = QPushButton("Export XML")
        self.generate_btn.clicked.connect(self._export_xml)
        layout.addWidget(self.generate_btn)

        self.output_label = QLabel("Output: extensions/multi-robot-array/output/robot_array.xml")
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet("color: #666;")
        layout.addWidget(self.output_label)

        layout.addStretch(1)

        return panel

    def _spinbox(self, min_val: float, max_val: float, value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(3)
        box.setRange(min_val, max_val)
        box.setSingleStep(1.0)
        box.setValue(value)
        return box

    def _pick_xml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select MuJoCo XML",
            str(Path.cwd()),
            "MuJoCo XML (*.xml);;All Files (*)",
        )
        if not path:
            return
        self.xml_path.setText(path)

    def _xml_path_changed(self) -> None:
        path = self.xml_path.text().strip()
        if not path:
            return

    def _load_default_xml(self) -> None:
        try:
            latest = self._generate._latest_robot_xml(Path(__file__).resolve().parents[2])
        except Exception:
            latest = None
        if latest:
            self.xml_path.setText(str(latest))

    def _preview_array(self) -> None:
        xml_path = self.xml_path.text().strip()
        if not xml_path:
            QMessageBox.warning(self, "Missing XML", "Please select an input XML file.")
            return
        input_xml = Path(xml_path)
        if not input_xml.exists():
            QMessageBox.warning(self, "Invalid XML", "Selected XML file does not exist.")
            return

        preview_dir = Path("extensions/multi-robot-array/output")
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_xml = preview_dir / "_preview_robot_array.xml"
        self._preview_xml_path = preview_xml

        try:
            self._generate.build_array(
                input_xml,
                preview_xml,
                self.offset_y.value(),
                self.count.value(),
                self.base_rot.value(),
                self.tilt_x.value(),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Preview Failed", f"Failed to generate preview XML:\n{exc}")
            return


    def _export_xml(self) -> None:
        xml_path = self.xml_path.text().strip()
        if not xml_path:
            QMessageBox.warning(self, "Missing XML", "Please select an input XML file.")
            return
        input_xml = Path(xml_path)
        if not input_xml.exists():
            QMessageBox.warning(self, "Invalid XML", "Selected XML file does not exist.")
            return

        output_xml = Path("extensions/multi-robot-array/output/robot_array.xml")
        try:
            self._generate.build_array(
                input_xml,
                output_xml,
                self.offset_y.value(),
                self.count.value(),
                self.base_rot.value(),
                self.tilt_x.value(),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Generate Failed", f"Failed to generate array XML:\n{exc}")
            return

        QMessageBox.information(self, "Generated", f"Wrote: {output_xml}")

    def _open_input_viewer(self) -> None:
        xml_path = self.xml_path.text().strip()
        if not xml_path:
            QMessageBox.warning(self, "Missing XML", "Please select an input XML file.")
            return
        input_xml = Path(xml_path)
        if not input_xml.exists():
            QMessageBox.warning(self, "Invalid XML", "Selected XML file does not exist.")
            return
        self._launch_viewer(input_xml)

    def _open_array_viewer(self) -> None:
        self._preview_array()
        if self._preview_xml_path is None or not self._preview_xml_path.exists():
            return
        self._launch_viewer(self._preview_xml_path)

    def _launch_viewer(self, xml_path: Path) -> None:
        script = (
            "import mujoco\n"
            "import mujoco.viewer\n"
            f"model = mujoco.MjModel.from_xml_path(r'''{xml_path}''')\n"
            "data = mujoco.MjData(model)\n"
            "mujoco.viewer.launch(model, data)\n"
        )
        proc = QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments(["-c", script])
        proc.start()
        self._viewer_processes.append(proc)



def main() -> None:
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    window = ArrayGui()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
