import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='netpaint',
    version='1.1.2',
    author='Anthony Westbrook',
    author_email='twestbrook@oursyntheticdreams.com',
    packages=setuptools.find_packages(),
    scripts=['netpaint.py'],
    url='http://github.com/SyntheticDreams/NetPaint',
    license='LICENSE',
    description='Terminal compatible text-based drawing program featuring full mouse support',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
    install_requires=[
        "urwid >= 2.0.1",
        "pillow >= 6.1.0",
    ],
)
