from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image
import tempfile
import os

class CropDialog(QtWidgets.QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.initUI()

    def initUI(self):
        # Set up the layout and image view
        self.setWindowTitle("Select Region to Crop")
        self.setGeometry(100, 100, 800, 600)
        
        self.graphicsView = QtWidgets.QGraphicsView(self)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.graphicsView.setScene(self.scene)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.graphicsView)

        # Load image into QGraphicsScene
        img_pixmap = QtGui.QPixmap.fromImage(self.pil_image_to_qimage(self.image))
        self.image_item = QtWidgets.QGraphicsPixmapItem(img_pixmap)
        self.scene.addItem(self.image_item)
        self.graphicsView.setScene(self.scene)

        # Set up the rectangle item for selection
        self.selection_rect = QtCore.QRectF()
        self.selection_item = QtWidgets.QGraphicsRectItem()
        self.selection_item.setPen(QtGui.QPen(QtCore.Qt.red, 2))
        self.scene.addItem(self.selection_item)

        # Connect dialog buttons
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout().addWidget(self.buttons)

        # Set up mouse event handling
        self.graphicsView.setMouseTracking(True)
        self.graphicsView.viewport().installEventFilter(self)

    def pil_image_to_qimage(self, pil_image):
        # Convert PIL Image to QImage
        img_byte_array = pil_image.convert('RGB').tobytes()
        return QtGui.QImage(img_byte_array, pil_image.width, pil_image.height, pil_image.width * 3, QtGui.QImage.Format_RGB888)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.LeftButton:
                self.start_pos = self.graphicsView.mapToScene(event.pos())
                self.selection_rect = QtCore.QRectF()
                self.selection_item.setRect(self.selection_rect)
        
        elif event.type() == QtCore.QEvent.MouseMove:
            if hasattr(self, 'start_pos'):
                end_pos = self.graphicsView.mapToScene(event.pos())
                self.selection_rect = QtCore.QRectF(self.start_pos, end_pos).normalized()
                self.selection_item.setRect(self.selection_rect)
        
        elif event.type() == QtCore.QEvent.MouseButtonRelease:
            if event.button() == QtCore.Qt.LeftButton:
                if not self.selection_rect.isNull():
                    self.accept()
        
        return super().eventFilter(obj, event)

    def get_crop_rect(self):
        # Return the selection rectangle
        return self.selection_rect
