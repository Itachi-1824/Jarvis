# 🤖 Jarvis - AI Personal Assistant

> *"Sometimes you gotta run before you can walk."* - Tony Stark

**Built by [Itachi-1824](https://github.com/Itachi-1824)**

## 📖 Overview

Jarvis is a sophisticated AI personal assistant inspired by Tony Stark's AI companion from Iron Man. This Python-based assistant leverages cutting-edge real-time AI technology to provide voice-interactive assistance with a touch of classy sarcasm and butler-like sophistication.

Powered by LiveKit's real-time communication platform and Google's advanced AI models, Jarvis can help you with weather updates, web searches, email management, and much more - all through natural voice conversations.

## ✨ Features

- 🎙️ **Real-time Voice Interaction** - Natural conversation with advanced speech recognition and synthesis
- 🌤️ **Weather Updates** - Get current weather information for any city worldwide
- 🔍 **Web Search** - Intelligent web search powered by DuckDuckGo
- 📧 **Email Management** - Send emails through Gmail with CC support
- 🎭 **Personality-driven AI** - Classy, sarcastic butler personality like Iron Man's Jarvis
- 🔊 **Noise Cancellation** - Enhanced audio quality with LiveKit's noise cancellation
- 📹 **Video Support** - Full video communication capabilities
- 🛡️ **Secure** - Environment-based configuration for API keys and credentials

## 🛠️ Technology Stack

- **Python 3.8+** - Core programming language
- **LiveKit** - Real-time communication platform
- **Google AI** - Advanced language model and voice synthesis
- **DuckDuckGo Search** - Privacy-focused web search
- **Gmail SMTP** - Email sending capabilities
- **dotenv** - Environment variable management
- **Requests** - HTTP client for API calls

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Gmail account with App Password enabled
- Google Cloud API key
- LiveKit account and credentials

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Itachi-1824/Jarvis.git
   cd Jarvis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file with your credentials.

## ⚙️ Configuration

Create a `.env` file in the project root with the following variables:

```env
# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-url.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Google AI Configuration
GOOGLE_API_KEY=your_google_api_key

# Gmail Configuration
GMAIL_USER=your_email@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password
```

### Getting Your Credentials

1. **LiveKit**: Sign up at [LiveKit Cloud](https://cloud.livekit.io/) and create a project
2. **Google AI**: Get your API key from [Google AI Studio](https://aistudio.google.com/)
3. **Gmail App Password**: Enable 2-factor authentication and generate an App Password

## 🎯 Usage

### Running the Assistant

```bash
python agent.py dev
```

The assistant will start and be ready to accept connections through LiveKit.
https://agents-playground.livekit.io/

### Interacting with Jarvis

Once connected, Jarvis will greet you and you can ask for:

- **Weather**: "What's the weather like in New York?"
- **Web Search**: "Search for the latest news about AI"
- **Email**: "Send an email to john@example.com about the meeting"

## 📁 Project Structure

```
Jarvis/
├── agent.py           # Main AI assistant implementation
├── prompts.py         # AI personality and instruction prompts
├── tools.py           # Tool functions (weather, search, email)
├── requirements.txt   # Python dependencies
├── .env.example      # Environment variables template
├── .gitignore        # Git ignore file
└── README.md         # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**"I am Iron Man's AI, but I can be yours too."** 🤖

Made with ❤️ by [Itachi-1824](https://github.com/Itachi-1824)

</div>
