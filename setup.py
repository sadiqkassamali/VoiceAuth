from setuptools import setup
import pypandoc
along_description = pypandoc.convert_file('README.md', 'rst')


setup(
    name='VoiceAuthreal',
    version='2.5',
    packages=[],
    url='https://github.com/sadiqkassamali/VoiceAuth',
    long_description=along_description,
    license='MIT',
    author='sadiq kassamali',
    author_email='Sadiqkassamali@gmail.com',
    description='Detect DeepFake or AI generated Audio and Authenticity',
)
