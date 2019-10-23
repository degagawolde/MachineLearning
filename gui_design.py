# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:57:03 2019

@author: Degaga
"""

import pandas as pd
import folium

from PyQt5 import QtWidgets, uic
from PyQt5 import QtWebEngineWidgets
from PyQt5.QtOpenGL import QGLWidget
from PyQt5 import QtGui
import sys

class Ui (QtWidgets.QMainWindow):
    def __init__(self):

        super(Ui, self).__init__()
        uic.loadUi('GUI.ui', self)
        self.load_file = self.findChild(QtWidgets.QPushButton, 'pushButton') # Find the button
        self.load_file.clicked.connect(self.fileLoadButtonPressed) # Remember to pass the definition/method, not the return value!
        
        self.train_model = self.findChild(QtWidgets.QPushButton, 'pushButton_2')
        self.train_model.clicked.connect(self.trainModelButtonPressed)
        
        self.train_result_text = self.findChild(QtWidgets.QTextBrowser,'textBrowser')
        
        self.train_result_graph = self.findChild(QtWidgets.QGraphicsView, 'graphicsView')
        self.train_result_graph.setViewport(QGLWidget())
        self.scene = QtWidgets.QGraphicsScene()
        
        
        self.table_widget = self.findChild(QtWidgets.QTableWidget, 'tableWidget')
        
        self.load_map = self.findChild(QtWidgets.QPushButton, 'pushButton_3') # Find the button
        self.load_map.clicked.connect(self.loadForestMap)
        
        self.input_file = self.findChild(QtWidgets.QLineEdit, 'lineEdit')
       
        self.webview = self.findChild(QtWebEngineWidgets.QWebEngineView, 'WebView')
        self.show()
        
    def trainModelButtonPressed(self):
        self.train_result_text.setText("training the model.....")     
        self.scene.addPixmap(QtGui.QPixmap('abay.jpg'))
        self.train_result_graph.setScene(self.scene)
        self.train_result_graph.show()
        self.show()
        
    def insertDataIntoTable(self):
        
        self.show()
        
    def fileLoadButtonPressed(self):
        # This is executed when the button is pressed
        filename, filetype=QtWidgets.QFileDialog.getOpenFileName()
        self.input_file.setText(filename)
        print(filename+" "+filetype)
        self.show()
        
    def loadForestMap(self):
        map_osm = folium.Map(location=[23.5747,58.1832],tiles='https://korona.geog.uni-heidelberg.de/tiles/roads/x={x}&y={y}&z={z}',attr= 'Imagery from <a href="http://giscience.uni-hd.de/">GIScience Research Group @ University of Heidelberg</a> &mdash; Map data &copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>')
        df = pd.read_csv("file.csv")
        for index, row in df.iterrows():
            folium.Marker(location=[row['Latitude'], row['Longitude']], popup=str(row['outlet_code']),icon=folium.Icon(color='red',icon='location', prefix='ion-ios')).add_to(map_osm)  

        map_osm.save('map.html')
        file= open("map.html", "r+")
        map_file=file.read();
        self.webview.setHtml(map_file)
        print(map_file)
        file.close()
        self.show()

app = QtWidgets.QApplication(sys.argv)
ap = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
win.setWindowTitle("win")
layout = QtWidgets.QVBoxLayout()

window = Ui()
app.exec_()