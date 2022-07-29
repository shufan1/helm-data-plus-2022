from distutils.core import setup

setup(
   name='utils',
   version='1.0',
   description='A useful module',
   author='Shufan Xia',
   author_email='sx78@duke.edu',
   packages=['utils'],  #same as name
   install_requires=['numpy', 'matplotlib', 'opencv-python','Pillow'], #external packages as dependencies
)


setup(
   name='preprocess',
   version='1.0',
   description='A useful module',
   author='Shufan Xia',
   author_email='sx78@duke.edu',
   packages=['preprocess'],  #same as name
   install_requires=['numpy', 'matplotlib', 'opencv-python','Pillow'], #external packages as dependencies
)
