from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from pyqtgraph.Qt import QtCore


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """

    sigDataChanged = QtCore.Signal()

    def __init__(
        self, defaultValues: Dict[str, Any], indexCol: str = None, parent=None
    ):
        super().__init__(parent)
        if indexCol is None:
            indexCol = next(iter(defaultValues.keys()))
        self.indexCol = indexCol
        self.defaultValues = defaultValues
        self.dataDf = pd.DataFrame(defaultValues, index=[indexCol])
        self.dataDf.drop(self.dataDf.index, inplace=True)

    def rowCount(self, parent=None):
        return self.dataDf.shape[0]

    def columnCount(self, parent=None):
        return self.dataDf.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            value = self.dataDf.iloc[index.row(), index.column()]
            if role == QtCore.Qt.DisplayRole:
                return str(value)
            elif role == QtCore.Qt.EditRole:
                return value
        return None

    def setData(
        self, index: QtCore.QModelIndex, value: Any, role: int = QtCore.Qt.ItemDataRole
    ) -> bool:
        super().setData(index, role)
        row = index.row()
        col = index.column()
        oldVal = self.dataDf.iat[row, col]
        # Try-catch for case of numpy arrays
        noChange = oldVal == value
        try:
            if noChange:
                return True
        except ValueError:
            # Happens with array comparison
            pass
        self.dataDf.iat[row, col] = value
        self.sigDataChanged.emit()
        return True

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.dataDf.columns[section]

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        base = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        # Allow renaming the "Layer Name" column
        if index.isValid() and "Layer Name" in self.defaultValues:
            nameCol = list(self.defaultValues.keys()).index("Layer Name")
            if index.column() == nameCol:
                return base | QtCore.Qt.ItemIsEditable
        return base

    def swapRows(self, row_a: int, row_b: int):
        """Swap two rows in the table (used for layer reordering)."""
        if not (0 <= row_a < self.rowCount() and 0 <= row_b < self.rowCount()):
            return
        self.layoutAboutToBeChanged.emit()
        idx_a = self.dataDf.index[row_a]
        idx_b = self.dataDf.index[row_b]
        row_a_data = self.dataDf.loc[idx_a].copy()
        self.dataDf.loc[idx_a] = self.dataDf.loc[idx_b]
        self.dataDf.loc[idx_b] = row_a_data
        self.layoutChanged.emit()
        self.sigDataChanged.emit()

    def addDfRows(self, keyValDict: Dict[str, List[Any]]):
        keyValDf = pd.DataFrame(keyValDict)
        defaultRows = self.makeDefaultTblRows(len(keyValDf))
        keyValDf.index = defaultRows.index
        defaultRows.update(keyValDf)
        self.layoutAboutToBeChanged.emit()
        self.dataDf = pd.concat([self.dataDf, defaultRows])
        self.layoutChanged.emit()

    def removeLayers(self, layers: Sequence[int] = None):
        if layers is None:
            layers = self.dataDf.index
        self.layoutAboutToBeChanged.emit()
        self.dataDf.drop(layers, inplace=True)
        self.layoutChanged.emit()

    def makeDefaultTblRows(self, numRows=1):
        nextNum = self.rowCount()
        outDict = {}
        for k, v in self.defaultValues.items():
            outDict[k] = [v] * numRows
        outDict[self.indexCol] = np.arange(nextNum, numRows)
        return pd.DataFrame(outDict).set_index(self.indexCol, drop=False)
