"""BoneHub Landmarks: a 3D Slicer module that loads a BoneHub landmarks file (landmarks-NN.zip, or its unpacked
landmarks-NN/ folder) into Slicer, in one subject hierarchy folder:

    point       a control point of the "points" list (one list for the file), labelled with the landmark's name
    points      a point list of its own
    line        a line markup
    cylinder    a line markup of its axis and a tube model of its radius
    circle      a closed curve model (or the centre point, when the source gives no normal)
    sphere      a sphere model and its centre point
    frame       a plane markup with the frame's axes
    area        areas.nrrd as a segmentation, <LANDMARK>.stl as a model

The coordinates in landmarks.json are LPS millimetres (see bonehub_format/landmarks.py); Slicer works in RAS, so x and
y change sign. Load the scan's image.nii.gz too: the landmarks are in its frame.

Install: Edit > Application Settings > Modules > Additional module paths, add this folder, restart Slicer. Then
Modules > BoneHub > BoneHub Landmarks, or from the Python console:

    slicer.util.getModuleLogic("BoneHubLandmarks").load("/path/to/landmarks-01.zip")
"""

import json
import os
import shutil
import tempfile
import zipfile

import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

LANDMARKS_JSON = "landmarks.json"
AREAS_FILE = "areas.nrrd"
PREFIX = "LANDMARK_"


class BoneHubLandmarks(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "BoneHub Landmarks"
        parent.categories = ["BoneHub"]
        parent.contributors = ["BoneHub"]
        parent.helpText = (
            "Loads a BoneHub landmarks file (landmarks-NN.zip or its folder): points, axes, spheres, frames and areas, "
            "in the frame of the scan's image."
        )
        parent.acknowledgementText = ""


class BoneHubLandmarksWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        form = qt.QFormLayout()
        self.layout.addLayout(form)
        import ctk

        self.pathEdit = ctk.ctkPathLineEdit()
        self.pathEdit.filters = ctk.ctkPathLineEdit.Files | ctk.ctkPathLineEdit.Dirs
        self.pathEdit.nameFilters = ["BoneHub landmarks (landmarks-*.zip landmarks.json)", "All files (*)"]
        form.addRow("Landmarks:", self.pathEdit)
        self.loadButton = qt.QPushButton("Load")
        form.addRow(self.loadButton)
        self.status = qt.QLabel("")
        self.status.wordWrap = True
        form.addRow(self.status)
        self.loadButton.connect("clicked(bool)", self.onLoad)
        self.layout.addStretch(1)

    def onLoad(self):
        path = self.pathEdit.currentPath
        if not path:
            self.status.text = "Choose a landmarks-NN.zip, its folder, or a landmarks.json."
            return
        try:
            nodes = BoneHubLandmarksLogic().load(path)
        except Exception as e:  # shown to the user, who can fix the file or path
            self.status.text = f"Could not load {os.path.basename(path)}: {e}"
            raise
        self.status.text = f"Loaded {len(nodes)} nodes from {os.path.basename(path)}."


def _ras(xyz):
    """LPS -> RAS."""
    return [-float(xyz[0]), -float(xyz[1]), float(xyz[2])]


def _label(entry):
    """The landmark's name without its prefix, with its rater, session and element where it has them."""
    name = entry["name"][len(PREFIX):] if entry["name"].startswith(PREFIX) else entry["name"]
    extra = [f"{k[0]}{entry[k]}" for k in ("rater", "session") if k in entry]
    if "element" in entry:
        extra.append(str(entry["element"]))
    return name + (f" [{' '.join(extra)}]" if extra else "")


def _description(entry):
    return json.dumps({k: v for k, v in entry.items() if k not in ("position", "positions", "points", "center", "axes", "origin")})


class BoneHubLandmarksLogic(ScriptedLoadableModuleLogic):
    def load(self, path):
        """Load a landmarks zip, folder or landmarks.json; return the nodes made."""
        path = os.path.abspath(path)
        temp = None
        try:
            if path.endswith(".zip"):
                temp = tempfile.mkdtemp(prefix="bonehub_landmarks_")
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist():
                        if os.path.basename(name) != name:
                            raise ValueError(f"{name} does not belong in a landmarks zip")
                    archive.extractall(temp)
                folder = temp
            elif os.path.isdir(path):
                folder = path
            else:
                folder = os.path.dirname(path)
            with open(os.path.join(folder, LANDMARKS_JSON), encoding="utf-8") as f:
                document = json.load(f)
            if document.get("coordinate_system") != "LPS" or document.get("units") != "mm":
                raise ValueError("landmarks.json must hold LPS coordinates in mm")
            title = os.path.splitext(os.path.basename(path.rstrip(os.sep)))[0]
            return self._load(document["landmarks"], folder, title)
        finally:
            if temp:
                shutil.rmtree(temp, ignore_errors=True)

    def _load(self, entries, folder, title):
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        parent = shNode.CreateFolderItem(shNode.GetSceneItemID(), title)
        nodes = []
        points = None
        areas_loaded = False

        def keep(node):
            shNode.SetItemParent(shNode.GetItemByDataNode(node), parent)
            nodes.append(node)
            return node

        for entry in entries:
            kind, label = entry["type"], _label(entry)
            if kind == "point":
                if points is None:
                    points = keep(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"{title} points"))
                    points.GetDisplayNode().SetTextScale(2.0)
                index = points.AddControlPoint(vtk.vtkVector3d(*_ras(entry["position"])), label)
                points.SetNthControlPointDescription(index, _description(entry))
            elif kind == "points":
                node = keep(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", label))
                for i, xyz in enumerate(entry["positions"], start=1):
                    node.AddControlPoint(vtk.vtkVector3d(*_ras(xyz)), f"{label} {i}")
                node.GetDisplayNode().SetPointLabelsVisibility(False)
            elif kind in ("line", "cylinder"):
                node = keep(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", label))
                for xyz in entry["points"]:
                    node.AddControlPoint(vtk.vtkVector3d(*_ras(xyz)))
                if kind == "cylinder":
                    keep(self._tube(entry["points"], entry["radius"], label))
            elif kind == "sphere":
                source = vtk.vtkSphereSource()
                source.SetCenter(*_ras(entry["center"]))
                source.SetRadius(float(entry["radius"]))
                source.SetThetaResolution(48)
                source.SetPhiResolution(24)
                source.Update()
                keep(self._model(source.GetOutput(), label, opacity=0.3))
                centre = keep(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"{label} centre"))
                centre.AddControlPoint(vtk.vtkVector3d(*_ras(entry["center"])), label)
            elif kind == "circle":
                if entry.get("normal"):
                    source = vtk.vtkRegularPolygonSource()
                    source.SetCenter(*_ras(entry["center"]))
                    source.SetNormal(*_ras(entry["normal"]))
                    source.SetRadius(float(entry["radius"]))
                    source.SetNumberOfSides(72)
                    source.GeneratePolygonOff()
                    source.Update()
                    keep(self._model(source.GetOutput(), label))
                centre = keep(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"{label} centre"))
                index = centre.AddControlPoint(vtk.vtkVector3d(*_ras(entry["center"])), label)
                centre.SetNthControlPointDescription(index, f"radius {entry['radius']} mm")
            elif kind == "frame":
                node = keep(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", label))
                axes = [_ras(axis) for axis in entry["axes"]]
                node.SetOrigin(_ras(entry["origin"]))
                node.SetAxes(axes[0], axes[1], axes[2])
                node.SetSize(20.0, 20.0)
            elif kind == "area":
                file = os.path.join(folder, entry["file"])
                if entry["file"] == AREAS_FILE:
                    if not areas_loaded:
                        keep(slicer.util.loadSegmentation(file, properties={"name": f"{title} areas"}))
                        areas_loaded = True
                else:
                    model = slicer.util.loadModel(file, properties={"coordinateSystem": "LPS"})
                    model.SetName(label)
                    keep(model)
            else:
                raise ValueError(f"unknown landmark type {kind!r}")
            if kind != "point" and nodes and hasattr(nodes[-1], "SetDescription"):
                nodes[-1].SetDescription(_description(entry))
        return nodes

    @staticmethod
    def _model(polydata, name, opacity=1.0):
        model = slicer.modules.models.logic().AddModel(polydata)
        model.SetName(name)
        model.GetDisplayNode().SetOpacity(opacity)
        return model

    def _tube(self, ends, radius, name):
        a, b = _ras(ends[0]), _ras(ends[1])
        line = vtk.vtkLineSource()
        line.SetPoint1(*a)
        line.SetPoint2(*b)
        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(float(radius))
        tube.SetNumberOfSides(48)
        tube.CappingOn()
        tube.Update()
        return self._model(tube.GetOutput(), f"{name} cylinder", opacity=0.3)
