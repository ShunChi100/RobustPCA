from distutils.core import setup

setup(
    name='RobustPCA',
    version='0.1dev',
    packages=['RobustPCA',],
    license='LICENSE.md',
    description='Robust Pincipal Component Analysis',
    long_description=open('README.md').read(),
    author = ['Shun Chi'],
    install_requires=[
        "numpy",
    ],
)
