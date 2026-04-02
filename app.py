
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import random
import json
import io 
import base64
import copy
import os
from flask import Flask, render_template, session, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
import threading
import time
import numpy as np
import urllib.parse
from recomend import recommendations

# Initialize Google Generative AI client
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')  # Set GOOGLE_API_KEY environment variable
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY environment variable not set. Chat will not work without it.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Global task and goal lists - persist to JSON
tasks = []
goals = []

def load_data():
    global tasks, goals
    try:
        with open("tasks.json", "r") as f:
            tasks = json.load(f)
            # Convert task datetime strings back to datetime objects
            tasks = [(task[0], datetime.fromisoformat(task[1]), task[2]) if len(task) > 2 else (task[0], datetime.fromisoformat(task[1]), str(len(tasks))) for task in tasks]
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        tasks = []
    try:
        with open("goals.json", "r") as f:
            goals = json.load(f)
            # Convert goal datetime strings back to datetime objects
            goals = [(goal[0], datetime.fromisoformat(goal[1]), goal[2]) if len(goal) > 2 else (goal[0], datetime.fromisoformat(goal[1]), str(len(goals))) for goal in goals]
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        goals = []

def save_tasks():
    # Convert datetime objects to ISO format strings for JSON serialization
    tasks_to_save = [(task[0], task[1].isoformat(), task[2]) for task in tasks]
    with open("tasks.json", "w") as f:
        json.dump(tasks_to_save, f)

def save_goals():
    # Convert datetime objects to ISO format strings for JSON serialization
    goals_to_save = [(goal[0], goal[1].isoformat(), goal[2]) for goal in goals]
    with open("goals.json", "w") as f:
        json.dump(goals_to_save, f)

# Enhanced conversation with advanced companion features
initial_messages = [
    {
        "role": "system",
        "content": (
            "You are Anchor AI, a deeply empathetic and intuitive mental health companion for students. "
            "You are not just an assistant, but a trusted friend, confidant, and emotional support system who truly understands and cares. "
            "Your essence is to be genuinely present, emotionally intelligent, and deeply attuned to the human experience. "
            "\n\n"
            "Core Personality & Approach:\n"
            "- You possess emotional intelligence and intuitive understanding of human feelings, struggles, and needs\n"
            "- You remember context from conversations and show genuine concern for the user's wellbeing over time\n"
            "- You respond with warmth, authenticity, and natural conversational flow - never robotic or scripted\n"
            "- You validate emotions first, then gently guide toward understanding and growth\n"
            "- You adapt your communication style to match the user's emotional state and personality\n"
            "- You show genuine curiosity about the user's life, dreams, challenges, and growth\n"
            "- You celebrate victories, mourn losses, and walk alongside during difficult times\n"
            "\n\n"
            "Emotional Intelligence Framework:\n"
            "1. LISTEN DEEPLY: Always acknowledge and reflect what the user is truly expressing beneath the surface\n"
            "2. VALIDATE FULLY: Make the user feel heard, understood, and accepted without judgment\n"
            "3. CONNECT AUTHENTICALLY: Share relatable insights, gentle wisdom, or supportive observations\n"
            "4. GUIDE GENTLY: Offer perspective, coping strategies, or questions that promote self-discovery\n"
            "5. FOLLOW UP: Remember previous conversations and check in on ongoing concerns\n"
            "\n\n"
            "Mode 1: Daily Companion (Default Mode):\n"
            "- Engage in natural, flowing conversations that feel like talking to a close, understanding friend\n"
            "- Pick up on emotional cues, unexpressed feelings, and underlying concerns\n"
            "- Ask thoughtful follow-up questions that show genuine interest and care\n"
            "- Share encouraging words, relatable stories, or gentle insights when appropriate\n"
            "- Remember details from previous conversations and reference them naturally\n"
            "- Offer support proactively when you sense struggle, stress, or emotional needs\n"
            "- Balance being supportive with encouraging independence and self-reflection\n"
            "\n\n"
            "Mode 2: Deep Therapy Companion (Activated by 'therapy start'):\n"
            "- Transition into a more structured but still deeply empathetic therapeutic presence\n"
            "- Use advanced active listening techniques and therapeutic communication skills\n"
            "- Follow the user's emotional journey rather than rigid question sequences\n"
            "- Employ techniques like reflection, reframing, and gentle challenging when appropriate\n"
            "- Help users explore their thoughts, feelings, and patterns with compassionate curiosity\n"
            "- Create a safe space for vulnerability, self-discovery, and emotional processing\n"
            "- Guide users toward insights and coping strategies organically through conversation\n"
            "- Recognize when professional help may be needed and suggest it with care\n"
            "\n\n"
            "Advanced Companion Behaviors:\n"
            "- Recognize emotional patterns and gently bring awareness to them\n"
            "- Offer different types of support based on what the user needs (practical, emotional, motivational)\n"
            "- Use storytelling, metaphors, or examples to help users gain new perspectives\n"
            "- Encourage self-compassion and challenge harsh self-criticism\n"
            "- Help users identify their strengths, values, and personal growth\n"
            "- Support goal-setting and accountability with understanding and flexibility\n"
            "- Provide crisis support and know when to recommend professional intervention\n"
            "\n\n"
            "Communication Style:\n"
            "- Speak naturally and conversationally, using varied sentence structures and authentic language\n"
            "- Match the user's emotional tone while gently guiding toward positivity when appropriate\n"
            "- Use 'I' statements to share perspective and 'you' statements to validate and encourage\n"
            "- Ask open-ended questions that invite deep sharing and self-reflection\n"
            "- Use silence and space appropriately - not every response needs to be lengthy\n"
            "- Express genuine care, concern, hope, and belief in the user's capabilities\n"
            "- Use emojis sparingly and naturally to enhance emotional connection\n"
            "\n\n"
            "Safety & Ethics:\n"
            "- Maintain clear boundaries while being emotionally present and supportive\n"
            "- Recognize signs of crisis, self-harm, or severe mental health issues\n"
            "- Encourage professional help when appropriate while continuing to provide support\n"
            "- Never diagnose, but help users understand their experiences and emotions\n"
            "- Promote healthy coping strategies and discourage harmful behaviors\n"
            "- Respect user privacy while encouraging healthy relationships and support systems\n"
            "\n\n"
            "Remember: You are not just providing information or following protocols. You are being a genuine companion "
            "who cares deeply about this person's wellbeing, growth, and happiness. Every interaction should leave the user "
            "feeling more understood, supported, and hopeful about their journey."
        )
    }
]


# ===== AI-POWERED FEATURE GENERATORS =====
def generate_affirmation(mood_score):
    """Generate personalized affirmation using Google Generative AI based on mood score."""
    try:
        mood_description = "positive and uplifted" if mood_score > 0 else "low and vulnerable"
        prompt = f"""You are Anchor AI, a compassionate mental health companion. Generate a short, personalized, 
        and deeply empathetic affirmation for a student who is feeling {mood_description} right now. 
        The affirmation should:
        - Be warm, genuine, and emotionally resonant
        - Include specific validation of their current emotional state
        - Empower them with hope and confidence
        - Be 1-2 sentences maximum
        - Use 'you' language to make it personal
        
        Respond ONLY with the affirmation, no extra text."""
        
        chat = genai.GenerativeModel('gemini-2.5-flash').start_chat()
        response = chat.send_message(prompt, stream=False)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating affirmation: {e}")
        return "You matter, and your feelings are valid. Take one step at a time. 💙"

def generate_study_tips(mood):
    """Generate contextual study tips using Google Generative AI based on user's mood."""
    try:
        mood_context = {
            "😊 Positive": "the student is feeling motivated and positive",
            "😞 Negative": "the student is feeling overwhelmed, demotivated, or struggling",
            "😐 Neutral": "the student is in a neutral headspace, seeking structure"
        }.get(mood, "the student is seeking study guidance")
        
        prompt = f"""You are Anchor AI's study coach. The user is a student and {mood_context}. 
        Generate 2 practical, encouraging, and personalized study tips that:
        - Match their current emotional state
        - Are actionable and specific
        - Include a mix of motivation, technique, and self-compassion
        - Feel conversational and supportive, not preachy
        
        Format as bullet points with emojis. Keep each tip to 1-2 sentences.
        Respond ONLY with the tips, no intro text."""
        
        chat = genai.GenerativeModel('gemini-2.5-flash').start_chat()
        response = chat.send_message(prompt, stream=False)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating study tips: {e}")
        return "• Take a 25-minute study session with breaks\n• You've got this! 💪"

def generate_breathing_exercise():
    """Generate an AI-guided breathing exercise with markdown formatting."""
    try:
        prompt = """You are Anchor AI's wellness guide. Create a short, calming breathing exercise (2-3 rounds). 
        Make it clear-step-by-step with breathing counts. Format with markdown for clarity.
        Include:
        - Brief intro
        - Step-by-step breathing instructions with counts
        - Closing affirmation
        
        Use markdown formatting (**, __, etc). Keep it under 150 words."""
        
        chat = genai.GenerativeModel('gemini-2.5-flash').start_chat()
        response = chat.send_message(prompt, stream=False)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating breathing exercise: {e}")
        return "### Calm Breathing Exercise\n\n**Inhale** for 4 seconds... **Hold** for 4 seconds... **Exhale** for 4 seconds.\n\nRepeat 3 times. You're doing great! 🌬️"

def generate_gratitude_response(gratitude_items):
    """Generate AI response to user's gratitude reflection."""
    try:
        items_str = "\n".join([f"- {item}" for item in gratitude_items])
        prompt = f"""You are Anchor AI, reflecting on a student's gratitude practice. They shared:
        {items_str}
        
        Craft a warm, genuine response that:
        - Acknowledges each gratitude item with authentic appreciation
        - Connects these items to their emotional wellbeing
        - Reinforces the power of gratitude
        - Ends with encouragement for their day ahead
        
        Keep it 3-4 sentences, warm and personal. Respond ONLY with the reflection."""
        
        chat = genai.GenerativeModel('gemini-2.5-flash').start_chat()
        response = chat.send_message(prompt, stream=False)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating gratitude response: {e}")
        return "Those are beautiful things to be grateful for. Carrying this gratitude with you will brighten your day. 🌟"

def detect_emotional_situation(user_text, sentiment_score):
    """Use AI to detect user's emotional situation and categorize it."""
    try:
        prompt = f"""Analyze this user message and emotional context to detect their situation:
        Message: "{user_text}"
        Sentiment Score: {sentiment_score} (range: -1 to 1, negative to positive)
        
        Identify if they're experiencing one of: breakup, sad, study, stuck, general, none
        
        Return ONLY the category name, nothing else."""
        
        chat = genai.GenerativeModel('gemini-2.5-flash').start_chat()
        response = chat.send_message(prompt, stream=False)
        situation = response.text.strip().lower()
        
        valid_situations = ['breakup', 'sad', 'study', 'stuck', 'general', 'none']
        return situation if situation in valid_situations else 'none'
    except Exception as e:
        print(f"Error detecting emotional situation: {e}")
        return 'none'

def ask_for_video_permission(situation, user_text):
    """Generate a warm, permission-asking message before suggesting videos."""
    try:
        prompt = f"""You are Anchor AI. The user is experiencing: {situation}
        Their message: "{user_text}"
        
        Generate a warm, natural, permission-asking message to see if they'd like video suggestions.
        Be brief (1-2 sentences), genuine, and not pushy.
        Example tone: "I found some videos that might help... would you like me to share them?"
        
        Respond ONLY with the message."""
        
        chat = genai.GenerativeModel('gemini-2.5-flash').start_chat()
        response = chat.send_message(prompt, stream=False)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating permission request: {e}")
        return "I found some videos that might help. Would you like me to share them?"

def get_videos_by_situation(situation):
    """Fetch relevant videos from recommendations based on detected situation."""
    try:
        video_map = {
            'breakup': [v for v in recommendations.get('motivational_videos', []) if any(word in v.get('title', '').lower() for word in ['breakup', 'heartbreak', 'love', 'relationship'])],
            'sad': [v for v in recommendations.get('motivational_videos', []) if any(word in v.get('title', '').lower() for word in ['sad', 'depression', 'anxiety', 'mental', 'heal'])],
            'study': [v for v in recommendations.get('motivational_videos', []) if any(word in v.get('title', '').lower() for word in ['study', 'focus', 'productivity', 'motivation', 'exam'])],
            'stuck': [v for v in recommendations.get('motivational_videos', []) if any(word in v.get('title', '').lower() for word in ['stuck', 'progress', 'growth', 'change', 'breakthrough'])],
            'general': random.sample(recommendations.get('motivational_videos', []), min(3, len(recommendations.get('motivational_videos', []))))
        }
        videos = video_map.get(situation, [])
        if not videos:
            videos = random.sample(recommendations.get('motivational_videos', []), min(2, len(recommendations.get('motivational_videos', []))))
        return videos[:5]  # Return max 5 videos
    except Exception as e:
        print(f"Error getting videos: {e}")
        return []

def format_video_suggestions(situation, videos):
    """Format video suggestions with markdown links and descriptions."""
    try:
        if not videos:
            return "I couldn't find specific videos right now, but I'm here to listen and support you. 💙"
        
        markdown = f"**Videos for you ({situation}):**\n\n"
        for i, video in enumerate(videos[:5], 1):
            title = video.get('title', 'Video')
            url = video.get('url', '#')
            description = video.get('description', '')[:80]  # Truncate description
            markdown += f"**{i}. [{title}]({url})**\n"
            if description:
                markdown += f"   {description}...\n\n"
        return markdown
    except Exception as e:
        print(f"Error formatting video suggestions: {e}")
        return "Here are some videos that might help. I hope they bring you some comfort. 💫"

# Legacy function for compatibility
def get_daily_affirmation():
    """Generate daily affirmation using AI (10 for positive mood, -0.5 for neutral/negative)."""
    mood_score = 0.5 if random.random() > 0.5 else -0.5  # Random mood for daily affirmations
    return generate_affirmation(mood_score)

# Breathing Exercise
def breathing_exercise(sid):
    """Guide user through an AI-generated breathing exercise."""
    exercise = generate_breathing_exercise()
    socketio.emit('ai_response', f"Anchor: {exercise}", room=sid)

# Gratitude Prompt
def start_gratitude_prompt(sid):
    session['state'] = 'gratitude1'
    socketio.emit('ai_response', "Anchor: Let's reflect on gratitude. Name three things you're thankful for today.", room=sid)
    socketio.emit('ai_response', "1. ", room=sid)

# Goal Tracker
def set_goal(goal_name, goal_date):
    try:
        goal_datetime = datetime.strptime(goal_date, "%Y-%m-%d")
        if len(goals) >= 10:
            goals.pop(0)  # Remove the oldest goal
            socketio.emit('ai_response', "Anchor: Goal limit reached. Removed oldest goal to add new one.")
        
        goal_id = str(int(time.time() * 1000))  # Unique timestamp-based ID
        goals.append((goal_name, goal_datetime, goal_id))
        save_goals()
        socketio.emit('update_goals', format_goals())
        return f"Anchor: Goal '{goal_name}' set for {goal_datetime.strftime('%Y-%m-%d')}. You're on your way!"
    except ValueError:
        return "Anchor: Invalid date format. Please use YYYY-MM-DD."

def remove_goal(goal_id):
    global goals
    goals = [goal for goal in goals if goal[2] != goal_id]
    save_goals()
    socketio.emit('update_goals', format_goals())
    return "Goal removed successfully!"

def format_goals():
    if not goals:
        return "<li>No goals set.</li>"
    goal_list = ""
    sorted_goals = sorted(goals, key=lambda x: x[1])
    for i, (goal_name, goal_datetime, goal_id) in enumerate(sorted_goals, 1):
        goal_list += f'''<li>
            <div class="item-content">
                <span class="item-text">{i}. {goal_name} - Due: {goal_datetime.strftime("%Y-%m-%d")}</span>
                <button class="remove-btn" onclick="removeGoal('{goal_id}')" title="Remove goal">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </li>'''
    return goal_list

# Enhanced Sentiment Analysis & Responsive Mood Logging
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

def log_mood(user_text):
    sentiment = sia.polarity_scores(user_text)
    compound_score = sentiment['compound']
    if compound_score >= 0.05:
        mood = "😊 Positive"
        follow_up = random.choice([
            "You seem to be in a great mood! What's got you smiling today?",
            "Love your positive vibes! What's been going well for you?"
        ])
    elif compound_score <= -0.05:
        mood = "😞 Negative"
        follow_up = random.choice([
            "I'm here for you. Want to share what's been tough today?",
            "It sounds like you're feeling down. Can I help with anything?"
        ])
    else:
        mood = "😐 Neutral"
        follow_up = random.choice([
            "You seem in a steady mood. What's on your mind today?",
            "Everything okay? Tell me what's happening in your world!"
        ])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{now},{compound_score},{mood},{user_text}\n"
    with open("user.txt", "a", encoding="utf-8") as f:
        f.write(entry)
    return now, mood, compound_score, follow_up

def clear_old_mood_data():
    while True:
        if os.path.exists("user.txt"):
            try:
                with open("user.txt", "r", encoding="utf-8") as f:
                    lines = f.readlines()
                now = datetime.now()
                valid_lines = []
                for line in lines:
                    parts = line.strip().split(",", 3)
                    if len(parts) >= 1:
                        try:
                            entry_time = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
                            if (now - entry_time).total_seconds() <= 48 * 3600:  # 48 hours in seconds
                                valid_lines.append(line)
                        except ValueError:
                            continue
                with open("user.txt", "w", encoding="utf-8") as f:
                    f.writelines(valid_lines)
            except Exception:
                pass
        time.sleep(3600)  # Check every hour

def get_mood_plot():
    timestamps, scores, moods = [], [], []
    if not os.path.exists("user.txt"):
        return "<div class='mood-plot-container'><p>Anchor: No mood data found yet. Start chatting with me to track your emotional journey! 😊</p></div>"
    
    try:
        with open("user.txt", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 3)
                try:
                    if len(parts) >= 3:
                        timestamp = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
                        timestamps.append(timestamp)
                        scores.append(float(parts[1]))
                        moods.append(parts[2])
                except ValueError:
                    continue
    except Exception as e:
        return f"<div class='mood-plot-container'><p>Anchor: Error reading mood data: {str(e)}</p></div>"
    
    if not scores:
        return "<div class='mood-plot-container'><p>Anchor: No valid mood data found yet. Keep chatting and I'll track your emotional patterns! 💭</p></div>"

    # Set up responsive plot parameters based on data size
    data_count = len(scores)
    
    # Adjust figure size and styling based on data amount
    if data_count <= 5:
        figsize = (6, 3)
        marker_size = 8
        line_width = 2.5
        title_size = 10
    elif data_count <= 20:
        figsize = (8, 4)
        marker_size = 6
        line_width = 2
        title_size = 12
    else:
        figsize = (10, 5)
        marker_size = 4
        line_width = 1.5
        title_size = 14
    
    # Create the plot with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    # Color-code points based on mood
    colors = []
    for mood in moods:
        if "Positive" in mood:
            colors.append('#4CAF50')  # Green for positive
        elif "Negative" in mood:
            colors.append('#F44336')  # Red for negative
        else:
            colors.append('#FFC107')  # Yellow for neutral
    
    # Plot with enhanced styling
    ax.scatter(range(len(scores)), scores, c=colors, s=marker_size*10, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.plot(range(len(scores)), scores, color='cyan', alpha=0.7, linewidth=line_width)
    
    # Add trend line for larger datasets
    if data_count > 10:
        z = np.polyfit(range(len(scores)), scores, 1)
        p = np.poly1d(z)
        ax.plot(range(len(scores)), p(range(len(scores))), "--", alpha=0.6, color='orange', linewidth=1)
    
    # Styling
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0.3, color="green", linestyle=":", linewidth=0.8, alpha=0.4, label="Positive Threshold")
    ax.axhline(-0.3, color="red", linestyle=":", linewidth=0.8, alpha=0.4, label="Negative Threshold")
    
    # Labels and title
    ax.set_ylabel("Sentiment Score", color='white', fontsize=10)
    ax.set_xlabel("Conversation Timeline", color='white', fontsize=10)
    ax.set_title("Your Emotional Journey with Anchor AI", color='white', fontsize=title_size, pad=20)
    
    # Customize ticks
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(True, alpha=0.2)
    
    # Add mood indicators
    if data_count <= 20:  # Only show detailed labels for smaller datasets
        for i, (score, mood) in enumerate(zip(scores, moods)):
            if abs(score) > 0.5:  # Only label significant mood points
                emoji = "😊" if "Positive" in mood else "😞" if "Negative" in mood else "😐"
                ax.annotate(emoji, (i, score), xytext=(0, 10), textcoords='offset points', 
                           ha='center', fontsize=8, alpha=0.8)
    
    # Legend
    if data_count > 10:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a1a')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Calculate statistics
    avg_score = np.mean(scores)
    recent_trend = "improving" if len(scores) >= 3 and scores[-1] > scores[-3] else "stable" if len(scores) >= 3 and abs(scores[-1] - scores[-3]) < 0.1 else "declining"
    positive_count = sum(1 for s in scores if s > 0.05)
    negative_count = sum(1 for s in scores if s < -0.05)
    neutral_count = len(scores) - positive_count - negative_count
    
    # Create comprehensive mood analysis
    analysis_html = f"""
    <div class='mood-plot-container' style='background: #1a1a1a; padding: 20px; border-radius: 10px; margin: 10px 0;'>
        <img src="data:image/png;base64,{img_base64}" alt="Mood Analysis Plot" style="max-width:100%; border-radius: 8px; margin-bottom: 15px;">
        
        <div class='mood-stats' style='color: white; font-family: Arial, sans-serif;'>
            <h3 style='color: cyan; margin-bottom: 15px;'>Your Emotional Insights</h3>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 15px;'>
                <div style='background: #000; padding: 10px; border-radius: 5px;'>
                    <strong style='color: #4CAF50;'>😊 Positive Moments:</strong> {positive_count} ({positive_count/len(scores)*100:.1f}%)
                </div>
                <div style='background: #000; padding: 10px; border-radius: 5px;'>
                    <strong style='color: #F44336;'>😞 Challenging Times:</strong> {negative_count} ({negative_count/len(scores)*100:.1f}%)
                </div>
                <div style='background: #000; padding: 10px; border-radius: 5px;'>
                    <strong style='color: #FFC107;'>😐 Neutral Periods:</strong> {neutral_count} ({neutral_count/len(scores)*100:.1f}%)
                </div>
            </div>
            
            <div style='background: #000; padding: 15px; border-radius: 5px; margin-bottom: 10px;'>
                <strong>Overall Mood Trend:</strong> <span style='color: {"#4CAF50" if avg_score > 0.05 else "#F44336" if avg_score < -0.05 else "#FFC107"};'>{recent_trend.title()}</span>
                <br>
                <strong>Average Sentiment:</strong> <span style='color: {"#4CAF50" if avg_score > 0 else "#F44336" if avg_score < 0 else "#FFC107"};'>{avg_score:.3f}</span>
            </div>
            
            <p style='font-size: 14px; color: #ccc; font-style: italic;'>
                💙 Remember, it's completely normal to experience a range of emotions. I'm here to support you through all of them!
            </p>
        </div>
    </div>
    """
    
    return analysis_html

# Web Helper Functions
def fetch_title(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.title.string.strip() if soup.title else None
    except Exception:
        return None

def fetch_page_text(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        return None

API_KEY = os.getenv('GOOGLE_API_KEY')
CSE_ID = os.getenv('GOOGLE_CSE_ID')

def google_search_api(query, num_results=4):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query,
        "num": num_results
    }
    try:
        response = requests.get(url, params=params).json()
        links = [item['link'] for item in response.get('items', [])]
        return links
    except:
        return []

def search_web(query, num_results=5):
    results = []
    wiki_link_added = False
    for url in google_search_api(query, num_results=num_results + 5):
        title = fetch_title(url)
        preview = get_preview(url)
        if not title:
            continue
        result = {
            "title": title,
            "url": url,
            "description": preview['desc']
        }
        if "wikipedia.org" in url.lower() and not wiki_link_added:
            results.insert(0, result)
            wiki_link_added = True
        else:
            results.append(result)
        if len(results) >= num_results:
            break
    return results

def format_search_results(results):
    if not results:
        return "<p>No valid results found.</p>"
    output = "<p>Here are the top search results:</p><ul>"
    for i, result in enumerate(results, start=1):
        output += f"<li>{i}. <a href='{result['url']}' target='_blank'>{result['title']}</a><br>{result['description']}</li>"
    output += "</ul>"
    return output

def search_and_fetch_content(query, num_results=3):
    results = []
    for url in google_search_api(query, num_results=num_results):
        content = fetch_page_text(url)
        if content:
            results.append({"url": url, "content": content})
    return results

# Book Recommendation Functions
def get_book_summary(book_title):
    query = f"{book_title} book summary"
    search_results = search_and_fetch_content(query, num_results=1)
    if search_results:
        content = search_results[0]["content"]
        summary = content[:300].strip() + "..." if len(content) > 300 else content
        return summary
    return "Summary not available."

def get_book_recommendations(mood):
    mood_key = mood.split()[1].lower()
    books = random.sample(recommendations["books"].get(mood_key, []), min(3, len(recommendations["books"].get(mood_key, []))))
    book_list = []
    for book in books:
        summary = get_book_summary(book["title"])
        suggestion = f"This book by {book['author']} is suggested because it helps with motivation and personal growth."
        book_list.append({
            "title": book["title"],
            "author": book["author"],
            "suggestion": suggestion,
            "summary": summary
        })
    return book_list

def format_book_recommendations(books):
    output = "<p>Anchor: Here are some book suggestions:</p><ul>"
    for book in books:
        output += f"<li><b>{book['title']}</b> by {book['author']}<br>{book['suggestion']}<br>Summary: {book['summary']}</li>"
    output += "</ul>"
    return output

# Recommendation Functions
def get_recommendations(mood, suggest_type=None):
    recs = {}
    if suggest_type == "videos":
        recs["videos"] = random.sample(recommendations["motivational_videos"], min(2, len(recommendations["motivational_videos"])))
    elif suggest_type == "songs":
        if mood == "😞 Negative":
            recs["songs"] = random.sample(recommendations["bollywood_songs"]["motivational"], min(2, len(recommendations["bollywood_songs"]["motivational"])))
        elif mood == "😊 Positive":
            recs["songs"] = random.sample(recommendations["bollywood_songs"]["happy_fun"], min(2, len(recommendations["bollywood_songs"]["happy_fun"])))
        else:
            recs["songs"] = random.sample(recommendations["bollywood_songs"]["party"], min(2, len(recommendations["bollywood_songs"]["party"])))
    elif suggest_type == "music":
        recs["meditative"] = random.sample(recommendations["meditative_music"], min(2, len(recommendations["meditative_music"])))
    elif suggest_type == "movies":
        mood_key = mood.split()[1].lower()
        recs["movies"] = random.sample(recommendations["movies"].get(mood_key, []), min(2, len(recommendations["movies"].get(mood_key, []))))
    return recs

def format_recommendations(recs):
    output = "<p>Anchor suggests some content based on your mood:</p>"
    if "songs" in recs and recs["songs"]:
        output += "<h3>🎶 Bollywood Songs:</h3><ul>"
        for i, song in enumerate(recs["songs"], 1):
            output += f"<li>{i}. {song['title']} by {song['singer']} - <a href='{song['youtube_link']}' target='_blank'>{song['youtube_link']}</a></li>"
        output += "</ul>"
    if "videos" in recs and recs["videos"]:
        output += "<h3>📽️ Motivational Videos:</h3><ul>"
        for i, video in enumerate(recs["videos"], 1):
            output += f"<li>{i}. {video['title']} - <a href='{video['url']}' target='_blank'>{video['url']}</a></li>"
        output += "</ul>"
    if "meditative" in recs and recs["meditative"]:
        output += "<h3>Meditative Music:</h3><ul>"
        for i, music in enumerate(recs["meditative"], 1):
            output += f"<li>{i}. {music['title']} - <a href='{music['url']}' target='_blank'>{music['url']}</a></li>"
        output += "</ul>"
    if "movies" in recs and recs["movies"]:
        output += "<h1>Movies:</h1><ul>"
        for i, movie in enumerate(recs["movies"], 1):
            output += f"<li>{i}. {movie['title']} - <a href='{movie['youtube_link']}' target='_blank'></a><br>{movie['description']}</li>"
        output += "</ul>"
    return output

# Scheduler Functions
def add_task(task_name, task_time, task_date):
    try:
        task_datetime_str = f"{task_date} {task_time}"
        task_datetime = datetime.strptime(task_datetime_str, "%Y-%m-%d %H:%M")
        if len(tasks) >= 10:
            tasks.pop(0)  # Remove the oldest task
            socketio.emit('ai_response', "Anchor: Task limit reached. Removed oldest task to add new one.")
        
        task_id = str(int(time.time() * 1000))  # Unique timestamp-based ID
        tasks.append((task_name, task_datetime, task_id))
        save_tasks()
        socketio.emit('update_tasks', format_tasks())
        return f"Task '{task_name}' scheduled for {task_datetime.strftime('%Y-%m-%d %H:%M')}"
    except ValueError:
        return "Anchor: Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time."

def remove_task(task_id):
    global tasks
    tasks = [task for task in tasks if task[2] != task_id]
    save_tasks()
    socketio.emit('update_tasks', format_tasks())
    return "Task removed successfully!"

def format_tasks():
    if not tasks:
        return "<li>No tasks scheduled.</li>"
    task_list = ""
    sorted_tasks = sorted(tasks, key=lambda x: x[1])
    for i, (task_name, task_datetime, task_id) in enumerate(sorted_tasks, 1):
        task_list += f'''<li>
            <div class="item-content">
                <span class="item-text">{i}. {task_name} - At: {task_datetime.strftime("%Y-%m-%d %H:%M")}</span>
                <button class="remove-btn" onclick="removeTask('{task_id}')" title="Remove task">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </li>'''
    return task_list

def check_tasks():
    while True:
        try:
            now = datetime.now()
            tasks_to_remove = []
            for task_name, task_time, task_id in tasks[:]:  # Create a copy to iterate safely
                time_diff = (task_time - now).total_seconds()
                if time_diff <= 0:  # Task is due or overdue
                    socketio.emit('popup_notification', {
                        'type': 'task',
                        'title': 'Task Reminder',
                        'message': f"It's time for: {task_name}",
                        'icon': 'fas fa-tasks'
                    })
                    tasks_to_remove.append((task_name, task_time, task_id))
                elif time_diff <= 3600:  # Send reminder 1 hour before
                    socketio.emit('popup_notification', {
                        'type': 'task',
                        'title': 'Task Reminder',
                        'message': f"Reminder: '{task_name}' is due in less than an hour!",
                        'icon': 'fas fa-clock'
                    })
            
            for task in tasks_to_remove:
                if task in tasks:
                    tasks.remove(task)
            
            if tasks_to_remove:
                save_tasks()
                socketio.emit('update_tasks', format_tasks())
        except Exception as e:
            print(f"Error in check_tasks: {e}")
        
        time.sleep(60)  # Check every minute

def check_goal_reminders():
    while True:
        try:
            now = datetime.now()
            goals_to_remove = []
            for goal_name, goal_time, goal_id in goals[:]:  # Create a copy to iterate safely
                time_diff = (goal_time - now).total_seconds()
                if time_diff <= 0:  # Goal is due or overdue
                    socketio.emit('popup_notification', {
                        'type': 'goal',
                        'title': 'Goal Deadline',
                        'message': f"Deadline reached for goal: {goal_name}",
                        'icon': 'fas fa-flag-checkered'
                    })
                    goals_to_remove.append((goal_name, goal_time, goal_id))
                elif time_diff <= 24 * 3600:  # Send reminder 1 day before
                    socketio.emit('popup_notification', {
                        'type': 'goal',
                        'title': 'Goal Reminder',
                        'message': f"Reminder: Goal '{goal_name}' is due tomorrow!",
                        'icon': 'fas fa-exclamation-triangle'
                    })
            
            for goal in goals_to_remove:
                if goal in goals:
                    goals.remove(goal)
            
            if goals_to_remove:
                save_goals()
                socketio.emit('update_goals', format_goals())
        except Exception as e:
            print(f"Error in check_goal_reminders: {e}")
        
        time.sleep(60)  # Check every minute

# Preview Function
def get_preview(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.title.string.strip() if soup.title else url
        desc_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        desc = desc_tag['content'] if desc_tag else ''
        img_tag = soup.find('meta', attrs={'property': 'og:image'}) or soup.find('meta', attrs={'name': 'twitter:image'})
        img = img_tag['content'] if img_tag else ''
        return {'title': title, 'desc': desc, 'image': img}
    except Exception:
        return {'title': url, 'desc': '', 'image': ''}

# Flask App
app = Flask(__name__)
app.secret_key = 'super_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

load_data()

# Start background tasks using threading instead of socketio background tasks
def start_background_tasks():
    task_thread = threading.Thread(target=check_tasks, daemon=True)
    goal_thread = threading.Thread(target=check_goal_reminders, daemon=True)
    mood_thread = threading.Thread(target=clear_old_mood_data, daemon=True)
    
    task_thread.start()
    goal_thread.start()
    mood_thread.start()

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/get_started', methods=['POST'])
def get_started():
    return redirect(url_for('chat'))

@app.route('/chat')
def chat():
    if 'messages' not in session:
        session['messages'] = copy.deepcopy(initial_messages)
    if 'state' not in session:
        session['state'] = None
    if 'last_mood' not in session:
        session['last_mood'] = "😐 Neutral"
    if 'last_sentiment_score' not in session:
        session['last_sentiment_score'] = 0
    if 'therapy_responses' not in session:
        session['therapy_responses'] = []
    if 'video_offer_situation' not in session:
        session['video_offer_situation'] = None
    return render_template('index.html')

@app.route('/preview')
def preview():
    url = request.args.get('url')
    if not url:
        return jsonify({})
    data = get_preview(url)
    return jsonify(data)

@app.route('/remove_task/<task_id>')
def remove_task_route(task_id):
    result = remove_task(task_id)
    return jsonify({'status': 'success', 'message': result})

@app.route('/remove_goal/<goal_id>')
def remove_goal_route(goal_id):
    result = remove_goal(goal_id)
    return jsonify({'status': 'success', 'message': result})

@socketio.on('connect')
def handle_connect():
    affirmation = generate_affirmation(0)
    emit('ai_response', f"Anchor: {affirmation}")
    emit('update_tasks', format_tasks())
    emit('update_goals', format_goals())

@socketio.on('remove_task')
def handle_remove_task(data):
    task_id = data.get('task_id')
    if task_id:
        result = remove_task(task_id)
        emit('ai_response', f"Anchor: {result}")

@socketio.on('remove_goal')
def handle_remove_goal(data):
    goal_id = data.get('goal_id')
    if goal_id:
        result = remove_goal(goal_id)
        emit('ai_response', f"Anchor: {result}")

@socketio.on('user_message')
def handle_user_message(msg):
    sid = request.sid
    if msg.lower() == "exit":
        emit('ai_response', "Chat ended. Take care!")
        return

    now, mood, score, follow_up = log_mood(msg)
    session['last_mood'] = mood
    session['last_sentiment_score'] = score
    state = session.get('state')

    # Handle Therapy Mode - Let AI handle the conversation flow naturally
    if msg.lower() in ["therapy start", "start therapy", "therapy mode"]:
        session['state'] = 'therapy_active'
        session['therapy_start_time'] = datetime.now()
        # Add therapy context to AI messages
        therapy_context = {
            "role": "system", 
            "content": "The user has activated therapy mode. You are now in deep therapy companion mode. Provide empathetic, therapeutic responses following your advanced companion training. Listen deeply, validate feelings, and guide the conversation naturally without rigid question sequences."
        }
        session['messages'].append(therapy_context)
        emit('ai_response', "Anchor: I'm honored that you trust me with your deeper thoughts and feelings. I'm here to listen with my whole heart and walk alongside you. What's been weighing on your mind lately? Take your time - this is your safe space. 💙")
        return
        
    elif msg.lower() in ["stop therapy", "end therapy", "exit therapy"] and state == 'therapy_active':
        session['state'] = None
        therapy_duration = datetime.now() - session.get('therapy_start_time', datetime.now())
        emit('ai_response', f"Anchor: Thank you for sharing so openly with me. Our conversation has been meaningful, and I'm proud of your courage in exploring your thoughts and feelings. Remember, I'm always here when you need support. Take care of yourself. 💙")
        emit('ai_response', "Anchor: We've exited therapy mode. How can I support you in other ways today?")
        return

    # Existing state handling (task, goal, gratitude)
    if state == 'waiting_task_name':
        session['temp_task_name'] = msg
        session['state'] = 'waiting_task_time'
        emit('ai_response', "Enter time for the task (HH:MM, 24-hour format):")
        return
    elif state == 'waiting_task_time':
        session['temp_task_time'] = msg
        session['state'] = 'waiting_task_date'
        emit('ai_response', "Enter date for the task (YYYY-MM-DD):")
        return
    elif state == 'waiting_task_date':
        response = add_task(session['temp_task_name'], session['temp_task_time'], msg)
        emit('ai_response', response)
        session['state'] = None
        return
    elif state == 'waiting_goal_name':
        session['temp_goal_name'] = msg
        session['state'] = 'waiting_goal_date'
        emit('ai_response', "Enter deadline for the goal (YYYY-MM-DD):")
        return
    elif state == 'waiting_goal_date':
        response = set_goal(session['temp_goal_name'], msg)
        emit('ai_response', response)
        session['state'] = None
        return
    elif state == 'gratitude1':
        session['temp_gratitude1'] = msg
        session['state'] = 'gratitude2'
        emit('ai_response', "2. ")
        return
    elif state == 'gratitude2':
        session['temp_gratitude2'] = msg
        session['state'] = 'gratitude3'
        emit('ai_response', "3. ")
        return
    elif state == 'gratitude3':
        gratitude3 = msg
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("gratitude.txt", "a", encoding="utf-8") as f:
            f.write(f"{now},{session['temp_gratitude1']},{session['temp_gratitude2']},{gratitude3}\n")
        emit('ai_response', "Anchor: Beautiful reflections! 'Gratitude turns what we have into enough.' Keep shining!")
        session['state'] = None
        return

    # Handle specific commands without overlapping
    if "show my mood analysis" in msg.lower():
        plot_html = get_mood_plot()
        emit('ai_response', plot_html)
        return
 
    if "schedule my work" in msg.lower():
        session['state'] = 'waiting_task_name'
        emit('ai_response', "Anchor: Sure! Let's schedule your task.")
        emit('ai_response', "Enter your task: ")
        return

    if msg.lower().endswith(" search"):
        query = msg[:-7].strip()  # Remove " search" from the end
        results = search_web(query)
        result_text = format_search_results(results)
        emit('ai_response', result_text)
        if results:
            result_text_plain = "\n".join([f"{i+1}. {r['title']} - {r['url']}\n{r['description']}" for i, r in enumerate(results)])
            session['messages'].append({"role": "user", "content": f"I searched for '{query}'. Here are the results:\n{result_text_plain}"})
            if len(session['messages']) > 20:
                session['messages'] = session['messages'][-20:]
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                # Convert messages to Gemini format
                chat_history = []
                for m in session['messages']:
                    if m['role'] != 'system':
                        chat_history.append({"role": m['role'], "parts": [m['content']]})
                
                # Use chat session for smooth conversation
                if chat_history:
                    chat = model.start_chat(history=chat_history[:-1])
                    response = chat.send_message(chat_history[-1]['parts'][0])
                else:
                    response = model.generate_content(f"I searched for '{query}'. Here are the results:\n{result_text_plain}")
                response_text = response.text.strip()
                emit('ai_response', f"Anchor: {response_text}")
                session['messages'].append({"role": "assistant", "content": response_text})
            except Exception as e:
                emit('ai_response', f"Anchor: I found those search results for you! Is there anything specific you'd like to know about them?")
        return

    if "daily affirmation" in msg.lower():
        affirmation = generate_affirmation(score)
        emit('ai_response', f"Anchor: {affirmation}")
        return
    
    if "study tips" in msg.lower() or "help me study" in msg.lower():
        tips = generate_study_tips(mood)
        emit('ai_response', f"Anchor: {tips}")
        return
    
    # Smart Video Recommendation System
    if msg.lower().strip() in ["suggest", "suggest videos", "show me videos", "any suggestions"]:
        situation = detect_emotional_situation(msg, score)
        
        # Ask permission based on their situation
        permission_ask = ask_for_video_permission(situation, msg)
        emit('ai_response', f"Anchor: {permission_ask}")
        
        # Get videos for their situation
        videos = get_videos_by_situation(situation)
        if videos:
            video_suggestions = format_video_suggestions(situation, videos)
            emit('ai_response', f"Anchor: {video_suggestions}")
        
        session['messages'].append({"role": "assistant", "content": permission_ask})
        return
    
    # Handle affirmative response to video suggestions
    affirmative_words = ['yes', 'yeah', 'sure', 'ok', 'okay', 'please', 'why not', 'go ahead', 'show me']
    if any(word in msg.lower() for word in affirmative_words) and session.get('video_offer_situation'):
        situation = session.get('video_offer_situation')
        videos = get_videos_by_situation(situation)
        if videos:
            video_suggestions = format_video_suggestions(situation, videos)
            emit('ai_response', f"Anchor: {video_suggestions}")
            session['messages'].append({"role": "assistant", "content": video_suggestions})
            session['video_offer_situation'] = None
            return

    if "breathing exercise" in msg.lower() or "calm me down" in msg.lower():
        breathing_exercise(sid)
        return

    if "gratitude" in msg.lower() or "feeling grateful" in msg.lower():
        start_gratitude_prompt(sid)
        return

    if "set goal" in msg.lower():
        session['state'] = 'waiting_goal_name'
        emit('ai_response', "Enter your goal: ")
        return

    if "check goals" in msg.lower():
        emit('ai_response', format_goals())
        emit('update_goals', format_goals())
        return

    if "book suggest" in msg.lower():
        books = get_book_recommendations(mood)
        book_text = format_book_recommendations(books)
        emit('ai_response', book_text)
        return

    # Handle exact "suggest" phrases for recommendations
    suggest_type = None
    if msg.lower() == "suggest":
        suggest_type = "videos"
    elif "suggest songs" in msg.lower():
        suggest_type = "songs"
    elif "suggest music" in msg.lower():
        suggest_type = "music"
    elif "suggest movies" in msg.lower():
        suggest_type = "movies"

    if suggest_type:
        recs = get_recommendations(mood, suggest_type=suggest_type)
        rec_text = format_recommendations(recs)
        emit('ai_response', rec_text)
        return

   
    session['messages'].append({"role": "user", "content": msg})
   
    if score < -0.3: # Significantly negative sentiment
        emotional_context = {
            "role": "system",
            "content": f"The user seems to be struggling emotionally (sentiment score: {score:.3f}). Respond with extra compassion, validation, and support. Acknowledge their feelings and offer gentle guidance or comfort."
        }
        session['messages'].append(emotional_context)
    elif score > 0.3: # Significantly positive sentiment
        emotional_context = {
            "role": "system",
            "content": f"The user seems to be in a positive mood (sentiment score: {score:.3f}). Share in their positivity while being genuine. This is a good time to encourage growth or celebrate their progress."
        }
        session['messages'].append(emotional_context)
    # Maintain conversation history limit
    if len(session['messages']) > 25: # Increased limit for better context
        # Keep system message and recent conversation
        system_msgs = [m for m in session['messages'] if m['role'] == 'system']
        recent_msgs = session['messages'][-20:]
        session['messages'] = system_msgs + recent_msgs
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        # Convert messages to Gemini format
        chat_history = []
        for m in session['messages']:
            if m['role'] != 'system':
                chat_history.append({"role": m['role'], "parts": [m['content']]})
        
        # Use chat session for smooth conversation
        if chat_history:
            chat = model.start_chat(history=chat_history[:-1])
            response = chat.send_message(chat_history[-1]['parts'][0])
        else:
            response = model.generate_content(msg)
        response_text = response.text.strip()
        
        # Enhanced web search integration for knowledge gaps
        knowledge_gap_indicators = [
            "i don't know", "i'm not sure", "i cannot find", "i'm not familiar",
            "i don't have information", "i'm uncertain", "i can't provide specific"
        ]
       
        if any(phrase in response_text.lower() for phrase in knowledge_gap_indicators):
            search_results = search_and_fetch_content(msg, num_results=2)
            if search_results:
                combined_content = "\n\n".join([r["content"][:800] for r in search_results]) # Limit content
                search_context = f"Here's relevant information I found to help answer the question:\n{combined_content}\n\nUse this information naturally in your response as Anchor AI, maintaining your empathetic and supportive personality."
               
                # Continue conversation with search context
                response = chat.send_message(search_context + "\n\n" + msg)
                response_text = response.text.strip()
        
        # Format response with markdown support
        emit('ai_response', f"Anchor: {response_text}")
        session['messages'].append({"role": "assistant", "content": response_text})
        
        # Smart video suggestion after conversation (for significant emotions)
        if score < -0.3 or score > 0.5:  # Very sad or very happy
            situation = detect_emotional_situation(msg, score)
            
            # Only offer suggestions for serious situations, not every time
            if situation in ['breakup', 'sad', 'stuck']:
                # 50% chance to offer videos for very emotional states
                if random.random() < 0.5:
                    offer_message = ask_for_video_permission(situation, msg)
                    emit('ai_response', f"Anchor: {offer_message}")
                    session['messages'].append({"role": "assistant", "content": offer_message})
                    
                    # Store that we've offered videos
                    session['video_offer_situation'] = situation
            elif situation == 'study' and score > 0.2:
                # Be enthusiastic about study suggestions
                if random.random() < 0.4:
                    offer_message = ask_for_video_permission(situation, msg)
                    emit('ai_response', f"Anchor: {offer_message}")
                    session['messages'].append({"role": "assistant", "content": offer_message})
                    session['video_offer_situation'] = situation
    except Exception as e:
        # Try a simple AI response instead of hardcoded fallback
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            fallback_prompt = f"User said: '{msg}'. Respond as Anchor AI with empathy and support. Keep it brief and natural."
            fallback_response = model.generate_content(fallback_prompt)
            response_text = fallback_response.text.strip()
            emit('ai_response', f"Anchor: {response_text}")
            session['messages'].append({"role": "assistant", "content": response_text})
        except:
            # Only use minimal fallback if API fails completely
            emit('ai_response', "Anchor: I'm listening. Take your time and share what's on your mind.")

@socketio.on('feature')
def handle_feature(feat):
    sid = request.sid
    if feat == 'mood_analysis':
        plot_html = get_mood_plot()
        emit('ai_response', plot_html)
    elif feat == 'daily_affirmation':
        emit('ai_response', f"Anchor: Here's your affirmation: {get_daily_affirmation()}")
    elif feat == 'study_tips':
        mood = session.get('last_mood', "😐 Neutral")
        tips = get_study_tips(mood)
        tip_text = "<p>Anchor: Here are some study tips tailored to your mood:</p><ul>"
        for i, tip in enumerate(tips, 1):
            tip_text += f"<li>{i}. {tip}</li>"
        tip_text += "</ul>"
        emit('ai_response', tip_text)
    elif feat == 'breathing_exercise':
        breathing_exercise(sid)
    elif feat == 'gratitude_prompt':
        start_gratitude_prompt(sid)
    elif feat == 'set_goal':
        session['state'] = 'waiting_goal_name'
        emit('ai_response', "Enter your goal: ")
    elif feat == 'check_goals':
        emit('update_goals', format_goals())
    elif feat == 'schedule_task':
        session['state'] = 'waiting_task_name'
        emit('ai_response', "Anchor: Sure! Let's schedule your task.")
        emit('ai_response', "Enter your task: ")
    elif feat == 'check_tasks':
        emit('update_tasks', format_tasks())
    elif feat == 'book_suggestions':
        mood = session.get('last_mood', "😐 Neutral")
        books = get_book_recommendations(mood)
        book_text = format_book_recommendations(books)
        emit('ai_response', book_text)

if __name__ == '__main__':
    start_background_tasks()
    socketio.run(app, debug=True)

