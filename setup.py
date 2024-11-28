from setuptools import setup

try:
    import pypandoc
    along_description = pypandoc.convert_file('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.MD').read()

setup(
    name='VoiceAuthreal',
    version='2.0',
    packages=[''],
    url='https://github.com/sadiqkassamali/VoiceAuth',
    long_description=along_description,
    license='Depends',
    author='sadiq kassamali',
    author_email='Sadiqkassamali@gmail.com',
    description='Detect DeepFake or AI generated Audio and Authencity',
)
