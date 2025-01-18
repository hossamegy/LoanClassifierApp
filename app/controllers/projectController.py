import os

class ProjectController:
  
  def __init__(self):
    self.base_path = os.path.dirname(os.path.dirname(__file__))
    
    