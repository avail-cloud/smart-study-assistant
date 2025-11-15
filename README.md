<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1VCwxTgsq6PIq3Zxwn85Iu0iDdfyS0ttM

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

# 🤖 AI-Powered Travel Assistant

Welcome to the AI-Powered Travel Assistant! This is a modern, multimodal web application designed to be your ultimate travel companion. Leveraging the power of the Google Gemini API, this assistant can answer your travel questions, identify landmarks from photos, create social media captions, analyze reviews, and even plan detailed, day-by-day itineraries based on your personal preferences.

![AI Travel Assistant Screenshot](https://storage.googleapis.com/aistudio-ux-team/prompts/12a.png)

## ✨ Features

- **Question Answering**: Ask any travel-related question and get instant, helpful answers.
- **Landmark Recognition**: Upload a photo of a landmark, and the AI will identify it and provide historical context.
- **Image Captioning**: Generate witty and engaging social media captions for your travel photos.
- **Text Summarization**: Paste a long travel guide or article to get a concise summary.
- **Sentiment Analysis**: Understand the sentiment (Positive, Negative, Neutral) of hotel or attraction reviews.
- **Detailed Itinerary Planning**: Specify your destination, dates, budget, and interests to receive a complete, day-by-day travel plan.
- **Multimodal Interaction**: Seamlessly switch between text-only and image-based queries.
- **Responsive Design**: A clean, mobile-friendly UI that works beautifully on any device.

## 🛠️ Tech Stack

- **Frontend**: React with TypeScript
- **Styling**: Tailwind CSS for a modern, utility-first design
- **AI Integration**: Google Gemini API (`@google/genai`)
- **Markdown Rendering**: `marked` library for rich text formatting in AI responses.

---

## ⚙️ Local Setup and Configuration

Follow these steps to get the Travel Assistant running on your local machine.

### 📋 Prerequisites

- **Node.js**: Make sure you have Node.js installed (version 18.x or higher is recommended). You can download it from [nodejs.org](https://nodejs.org/).
- **npm**: A package manager for JavaScript, which comes bundled with Node.js.
- **Gemini API Key**: You need an API key from Google AI Studio to use the application.

### Step 1: Clone the Repository

First, clone the project files to your local machine using Git.

```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Install Dependencies

Install all the necessary project dependencies using npm.

```bash
npm install
```

### Step 3: Environment Configuration (.env file)

This is a crucial step for connecting the application to the Gemini API.

1.  **Create a `.env` file** in the root directory of the project. This file will store your secret API key.

    ```bash
    # For Windows Command Prompt
    copy .env.example .env

    # For Windows PowerShell or other terminals (Git Bash, etc.)
    cp .env.example .env
    ```
    *If an `.env.example` file is not provided, simply create a new file named `.env`.*

2.  **Add your API Key**: Open the newly created `.env` file and add your Google Gemini API key as shown below.

    **File: `.env`**
    ```
    # Get your API key from Google AI Studio: https://aistudio.google.com/app/apikey
    API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```

    Replace `YOUR_GEMINI_API_KEY_HERE` with your actual key.

### Step 4: Run the Application

Once the dependencies are installed and the `.env` file is configured, you can start the local development server.

```bash
npm run dev
```

This command will launch the application. You can now open your web browser and navigate to the local address provided in the terminal (usually `http://localhost:5173` or a similar port).

## ❓ FAQ

### Is the `.env` file necessary for the local setup?

**Yes, absolutely.** The `.env` file is essential for a local setup for two main reasons:
1.  **Security**: It keeps your API key secret and separate from your source code. You should never commit API keys directly into your code or share them in a public repository. The `.gitignore` file is typically configured to ignore `.env` files for this reason.
2.  **Configuration**: The application code (`services/geminiService.ts`) is written to look for an environment variable named `API_KEY`. The `.env` file is the standard way to provide this variable to the application in a local development environment.

### Where do I get a Gemini API Key?

You can get a free API key from **[Google AI Studio](https://aistudio.google.com/app/apikey)**. Sign in with your Google account, create a new API key, and copy it into your `.env` file.

## 🕹️ How to Use the App

1.  **Select a Feature**: At the top of the input area, click on the feature you want to use (e.g., `Question Answering`, `Identify Landmark`, `Plan Itinerary`).
2.  **Provide Input**:
    - For text-based features, type your query into the text box.
    - For image-based features, click the image icon to upload a photo.
    - For the Itinerary Planner, fill out the form with your travel details.
3.  **Send**: Click the "Send" or "Generate Itinerary" button.
4.  **View Response**: The assistant's response will appear in the chat window above. Itineraries and other formatted responses will be displayed with clear styling.
