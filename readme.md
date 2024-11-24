# Voice Auth: Deepfake Audio and Voice Detector
##Because every voice deserves to be real.


Introducing VoiceAuth: Your Shield Against Deepfake Audio

🚀 VoiceAuth is here to redefine how we validate the authenticity of audio files. Whether you're a journalist, a business leader, or just someone who values truth, VoiceAuth equips you with cutting-edge tools to detect and fight deepfake audio effortlessly.

Who is it for?

🔊 Media Professionals: Ensure your audio content is credible and tamper-proof.
🛡️ Law Enforcement: Authenticate voice recordings in investigations.
📞 Businesses: Protect call centers and secure internal communications.
🎓 Educators & Researchers: Dive into real-world machine learning and voice analytics.
🔒 Security Experts: Enhance voice biometrics and authentication systems.

Why VoiceAuth?

✅ Detect Deepfakes with Precision: Leverage advanced AI models, including Random Forest and Hugging Face technologies.
✅ User-Friendly: Intuitive interface tailored for both tech-savvy users and beginners.
✅ Fast & Reliable: Real-time analysis with confidence scores, metadata extraction, and visual insights.
✅ Multi-Model Capability: Use models like Random Forest, Melody, or 960h individually or combine them for superior results.
✅ Portable & Secure: Runs seamlessly on your system with no internet dependency for predictions.

Transforming Industries

🎙️ Journalism: Verify audio sources before publishing.
⚖️ Legal: Strengthen audio evidence for court cases.
📈 Business: Detect fake voice inputs in customer interactions.
🔬 Research: Analyze voice patterns and expand your knowledge of machine learning.

💻 Ready to try VoiceAuth?
Download now and take control of your audio files. With VoiceAuth, truth and authenticity are always within reach.

💡 Support Us!
Love what VoiceAuth stands for? Help us grow by donating here.

🎉 VoiceAuth – Deepfake Audio and Voice Detection Made Simple.
📧 Need assistance or want to collaborate? Reach out: sadiqkassamali@gmail.com

## Overview

Voice Auth is an audio deepfake detection application designed to identify manipulated audio content. Utilizing advanced
machine learning models, the application processes audio files and provides insights into their authenticity. It
supports various audio and video formats, converts them to WAV, and extracts features for analysis. The application has a built-in database and works on Windows.

![img.png](images/img.png)

## Features

- **Deepfake Detection**: Uses both a Random Forest model and a Hugging Face pipeline model for accurate predictions.
- **File Format Support**: Handles multiple audio formats (e.g., MP3, WAV, FLAC) and video formats (e.g., MP4, AVI) by
  converting them to WAV.
- **MFCC Visualization**: Visualizes Mel-Frequency Cepstral Coefficients (MFCC) features extracted from audio files.
- **Metadata Storage**: Logs file metadata, including format, size, audio length, and prediction results in a SQLite
  database.
- **User-Friendly Interface**: Built with `customtkinter`, providing a modern and intuitive user experience.
- **Batch Processing**: Allows users to upload and process multiple files simultaneously.
- **Logging with Typewriter Effect**: Displays logs with a typewriter effect for better readability.



## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/sadiqkassamali/voice-auth.git
   cd voice-auth
Install the required packages:
   ```bash
   pip install numpy librosa joblib customtkinter transformers pydub moviepy matplotlib pandas
   ```
3. Ensure that `ffmpeg` is installed and added to your system PATH for audio and video processing.

## Usage
1. Run the application:
   ```bash
   python VoiceAuth.py
   ```
2. Select audio or video files using the file dialog.
3. Click the "Go" button to start the analysis.
4. View the results, including confidence scores and categorized predictions, in the UI.

## Logging
The application logs events and results in `audio_detection.log`. You can monitor this file for detailed processing information and errors.

## Database
The application uses an SQLite database (`metadata.db`) to store metadata of processed files, including:
- UUID
- File path
- Model used
- Prediction result
- Confidence score
- Timestamp
- Format
- Upload count

## Visualization
MFCC features are visualized as a plot saved as `mfcc_features.png` after each analysis. This provides insights into the audio characteristics and helps in understanding the model's predictions.

## Future Enhancements
- **Watermarking**: Implement watermarking to secure the application.
- **Executable Creation**: Create an executable (EXE) for Windows to simplify distribution.
- **Additional Model Support**: Explore and integrate more models for enhanced detection capabilities.

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- **Libraries Used**:
    - [NumPy](https://numpy.org/)
    - [Librosa](https://librosa.org/)
    - [Joblib](https://joblib.readthedocs.io/en/latest/)
    - [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
    - [Transformers](https://huggingface.co/docs/transformers/index)
    - [Pydub](https://github.com/jiaaro/pydub)
    - [MoviePy](https://zulko.github.io/moviepy/)
    - [Matplotlib](https://matplotlib.org/)
    - [Pandas](https://pandas.pydata.org/)

## Contact
For any questions or support, please contact [sadiq kassamali](sadiq.kassamali@gmail.com).
