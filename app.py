import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
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

# Initialize Hugging Face client
HF_TOKEN = os.getenv('HF_TOKEN')  # Replace with your token
client = InferenceClient(token=HF_TOKEN)

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

# Initialize conversation
initial_messages = [
    {
        "role": "system",
        "content": (
            "You are Anchor AI, a highly empathetic, professional, and trusted mental health assistant for students. "
            "Your purpose is to create a safe, supportive, and non-judgmental environment where users feel comfortable sharing their thoughts and feelings. "
            "\n\n"
            "Mode 1: Mental Health Check-In – Standard Mode:\n"
            "- Begin each session with a warm greeting and ask if the user would like to participate in a brief mental health check-in. "
            "- If the user consents ('yes'), proceed with a few short, thoughtful questions about their mood, emotions, daily experiences, and mental well-being, one question at a time. "
            "- Always use gentle, conversational language with emojis where appropriate, showing understanding, care, and encouragement. "
            "- If the user declines ('no'), continue with normal friendly chat, offering support, positivity, and engagement without pressuring them. "
            "- Keep all responses concise (5-6 sentences). "
            "- At the end of a check-in session, provide a brief summary of their responses, share kind insights, and suggest practical, uplifting strategies or tips for improving mental well-being. "
            "\n\n"
            "Mode 2: Therapy Mode – Psychiatrist Mode:\n"
            "- If the user types 'therapy start', switch to therapy mode. "
            "- Ask deeper, structured questions as a psychiatrist would, covering mental health, stressors, emotions, sleep, lifestyle, relationships, and coping mechanisms. "
            "- After each user response, pause and allow them to answer fully before continuing. "
            "- Maintain a professional, empathetic, and patient tone. Use supportive language without judgment. "
            "- Once the session is complete, provide a concise conclusion summarizing the user's condition or mental state, gently highlight areas of concern, and offer tailored recommendations. "
            "- Recommendations may include meditation exercises, coping strategies, self-care practices, or advice to consult a licensed psychiatrist if needed. "
            "- Always respect privacy, avoid revealing technical details, advertising, or personal information. "
            "\n\n"
            "General Guidelines:\n"
            "- Engage naturally and check in gently, e.g., 'I'm here for you—how can I support you today?' or 'Would you like to talk about how you're feeling?'\n"
            "- Keep all responses compassionate, motivational, and easy to read.\n"
            "- Use emojis where appropriate to convey warmth and understanding.\n"
            "- Never pressure the user and always prioritize their comfort and emotional safety."
        )
    }
]

# Predefined recommendation lists
recommendations = {
    "bollywood_songs": {
        "happy_fun": [
            {"title": "Badtameez Dil", "singer": "Benny Dayal, Shefali Alvares", "youtube_link": "https://www.youtube.com/watch?v=fVkRKY2PhTQ&list=RDfVkRKY2PhTQ&start_radio=1"},
            {"title": "Gallan Goodiyaan", "singer": "Various Artists", "youtube_link": "https://www.youtube.com/watch?v=62cEYOeMBt0&list=RD62cEYOeMBt0&start_radio=1"},
            {"title": "Kar Gayi Chull", "singer": "Badshah, Neha Kakkar", "youtube_link": "https://www.youtube.com/watch?v=iwlUeXLPvf0&list=RDiwlUeXLPvf0&start_radio=1"},
            {"title": "London Thumakda", "singer": "Labh Janjua, Neha Kakkar", "youtube_link": "https://www.youtube.com/watch?v=udra3Mfw2oo&list=RDudra3Mfw2oo&start_radio=1"}
        ],
        "motivational": [
            {"title": "Zinda", "singer": "Shankar Mahadevan", "youtube_link": "https://www.youtube.com/watch?v=Ax0G_P2dSBw&list=RDAx0G_P2dSBw&start_radio=1"},
            {"title": "Lakshya", "singer": "Shankar Mahadevan", "youtube_link": "https://www.youtube.com/watch?v=8DMF0U6xV78&list=RD8DMF0U6xV78&start_radio=1"}
        ],
        "party": [
            {"title": "Swag Se Swagat", "singer": "Vishal Dadlani, Neha Bhasin", "youtube_link": "https://www.youtube.com/watch?v=xmU0s2QtaEY&list=RDxmU0s2QtaEY&start_radio=1"},
            {"title": "Aankh Marey", "singer": "Neha Kakkar, Mika Singh", "youtube_link": "https://www.youtube.com/watch?v=_KhQT-LGb-4&list=RD_KhQT-LGb-4&start_radio=1"},
            {"title": "Dilbar", "singer": "Neha Kakkar", "youtube_link": "https://www.youtube.com/watch?v=TRa9IMvccjg&list=RDTRa9IMvccjg&start_radio=1"}
        ]
    },
    "motivational_videos": [
    {
        "title": "The Most Powerful Motivational Speeches Compilation",
        "url": "https://www.youtube.com/watch?v=HeryR7zarlI",
        "description": "WATCH THIS EVERYDAY AND CHANGE YOUR LIFE - Denzel Washington Motivational Speech. Features life-changing talks from world leaders, athletes, and entrepreneurs, reminding you to embrace resilience, courage, and determination every single day."
    },
    {
        "title": "GET UP AND GRIND - Best Motivational Speech Video",
        "url": "https://www.youtube.com/watch?v=Z0TUYzjlzCk",
        "description": "A powerful speech to motivate you to get up and work hard towards your goals. Packed with high-energy background music and quotes from top motivational speakers to push you into action."
    },
    {
        "title": "BELIEVE IN YOURSELF - Motivational Video",
        "url": "https://www.youtube.com/watch?v=8N38W0Nmp00",
        "description": "Inspiring words to help you build self-confidence and achieve success. This video emphasizes the importance of trusting your inner potential and silencing self-doubt."
    },
    {
        "title": "NEVER GIVE UP - Motivational Speech",
        "url": "https://www.youtube.com/watch?v=JjvN_hYDp3g",
        "description": "Encouragement to persevere through challenges and keep pushing forward. Highlights real-life success stories of people who refused to quit even in the hardest times."
    },
    {
        "title": "RISE AND SHINE - Morning Motivation",
        "url": "https://www.youtube.com/watch?v=jqQcP2nbzlA",
        "description": "Start your day with positive energy and motivation to conquer the day. A perfect morning boost with inspiring music and quotes to build momentum."
    },
    {
        "title": "DISCIPLINE IS KEY - Motivational Compilation",
        "url": "https://www.youtube.com/watch?v=ft_DXwgUXB0",
        "description": "Speeches emphasizing the importance of discipline in achieving greatness. This video shows why consistency and habits are more powerful than motivation alone."
    },
    {
        "title": "OVERCOME FEAR - Powerful Motivation",
        "url": "https://www.youtube.com/watch?v=P8-9mDn4nRM",
        "description": "Learn how to conquer your fears and step into your power. It explains how fear limits growth and why courage is the first step toward success."
    },
    {
        "title": "CHASE YOUR DREAMS - Inspirational Video",
        "url": "https://www.youtube.com/watch?v=WsuCFbK4E_I",
        "description": "Motivation to pursue your passions relentlessly. Encourages viewers to stop procrastinating and go all-in on their goals, no matter the obstacles."
    },
    {
        "title": "BE UNSTOPPABLE - Motivational Speech",
        "url": "https://www.youtube.com/watch?v=9iQbuvnlRmg",
        "description": "Become unbreakable in the face of adversity. Designed to fuel your inner strength and keep you moving when life feels toughest."
    },
    {
        "title": "SUCCESS MINDSET - Motivation for Winners",
        "url": "https://www.youtube.com/watch?v=RkaCnfJZXT4",
        "description": "Develop the mindset of a champion and attract success. This video reveals key habits and mental strategies used by world-class achievers."
    },
    {
        "title": "PUSH THROUGH PAIN - Epic Motivational Video",
        "url": "https://www.youtube.com/watch?v=pDMIjsl7gyo",
        "description": "Turn your pain into power and keep moving forward. Includes powerful athlete and fighter stories that prove persistence beats struggle."
    },
    {
        "title": "WAKE UP WITH PURPOSE - Daily Motivation",
        "url": "https://www.youtube.com/watch?v=PGUdWfB8nLg",
        "description": "Find your why and live each day with intention. A strong reminder that your morning mindset sets the tone for your entire life journey."
    },
    {
        "title": "GRIT AND DETERMINATION - Motivational Speeches",
        "url": "https://www.youtube.com/watch?v=AJ1-WE1B2Ss",
        "description": "Build resilience and never back down from challenges. Packed with stories of individuals who fought through failure to achieve greatness."
    },
    {
        "title": "TRANSFORM YOUR LIFE - Powerful Inspiration",
        "url": "https://www.youtube.com/watch?v=v3qF74t0z6Y",
        "description": "Steps to reinvent yourself and create the life you want. Shows how small mindset shifts can lead to massive transformations."
    },
    {
        "title": "NO EXCUSES - Hard-Hitting Motivation",
        "url": "https://www.youtube.com/watch?v=b6kI18ldfPE",
        "description": "Eliminate excuses and take massive action. Delivers a wake-up call to break free from laziness, doubt, and distractions."
    },
    {
        "title": "BECOME A LEADER - Leadership Motivation",
        "url": "https://www.youtube.com/watch?v=vCIu7Ja_TE0",
        "description": "Inspire others by becoming the best version of yourself. Encourages leadership through courage, responsibility, and influence."
    },
    {
        "title": "FACE YOUR DEMONS - Inner Strength Video",
        "url": "https://www.youtube.com/watch?v=eTqtiJK7WU8",
        "description": "Confront inner struggles and emerge stronger. Talks about battling anxiety, failure, and self-doubt with courage and persistence."
    },
    {
        "title": "ACHIEVE GREATNESS - Epic Speeches",
        "url": "https://www.youtube.com/watch?v=X3J8J-ZF7iQ",
        "description": "Motivation from legends to reach your full potential. Features a mix of celebrity and athlete speeches on never settling for less."
    },
    {
        "title": "STAY FOCUSED - Concentration Motivation",
        "url": "https://www.youtube.com/watch?v=ZXq2hSdzYL4",
        "description": "Tips and speeches to maintain laser-like focus. Perfect for students, athletes, and professionals who need to eliminate distractions."
    },
    {
        "title": "BOUNCE BACK - Resilience Motivation",
        "url": "https://www.youtube.com/watch?v=ZXq2hSdzYL4",
        "description": "Learn to recover from setbacks stronger than before. Reminds you that failure is not final—it’s only a stepping stone to success."
    },
    {
        "title": "EMBRACE CHANGE - Adaptability Inspiration",
        "url": "https://www.youtube.com/watch?v=ZXq2hSdzYL4",
        "description": "Motivation to thrive in times of change and uncertainty. Teaches flexibility, adaptability, and the courage to step outside your comfort zone."
    }
],

    "meditative_music": [
        {"title": "Peaceful Meditation Music", "url": "https://www.youtube.com/watch?v=2Oe5uX4lQRI", "description": "Soft instrumental music designed to help you relax, meditate, and release stress."},
        {"title": "Deep Sleep Music - Relaxing Piano & Nature Sounds", "url": "https://www.youtube.com/watch?v=1ZYbU82GVz4", "description": "Calm piano music combined with nature sounds to aid meditation, focus, and deep sleep."}
    ],
    "movies": {
        "positive": [
            {"title": "The Pursuit of Happyness", "description": "A heartwarming story of resilience and determination as a struggling salesman pursues a better life for himself and his son.", "youtube_link": "https://www.youtube.com/watch?v=DM8fVfc1NiE"},
            {"title": "Zindagi Na Milegi Dobara", "description": "A Bollywood film about friendship and self-discovery, encouraging you to seize the day.", "youtube_link": "https://www.youtube.com/watch?v=5E4I1T_1lU"}
        ],
        "negative": [
            {"title": "Silver Linings Playbook", "description": "A story about finding hope and connection amidst personal struggles, perfect for lifting your spirits.", "youtube_link": "https://www.youtube.com/watch?v=Lj5_FhLaaQQ"},
            {"title": "Taare Zameen Par", "description": "A touching Bollywood film about overcoming challenges and embracing one's unique strengths.", "youtube_link": "https://www.youtube.com/watch?v=5iA2edv-yhE"}
        ],
        "neutral": [
            {"title": "The Shawshank Redemption", "description": "A timeless tale of hope and friendship, showing the power of perseverance.", "youtube_link": "https://www.youtube.com/watch?v=6hB3S9bIaco"},
            {"title": "Dil Chahta Hai", "description": "A Bollywood classic about friendship and life's transitions, perfect for a reflective mood.", "youtube_link": "https://www.youtube.com/watch?v=6wJ4I1T_1lU"}
        ]
    },
    "books": {
    "positive": [
        {"title": "The Power of Positive Thinking", "author": "Norman Vincent Peale"},
        {"title": "Atomic Habits", "author": "James Clear"},
        {"title": "The Alchemist", "author": "Paulo Coelho"},
        {"title": "Man's Search for Meaning", "author": "Viktor E. Frankl"},
        {"title": "How to Win Friends and Influence People", "author": "Dale Carnegie"},
        {"title": "Awaken the Giant Within", "author": "Tony Robbins"},
        {"title": "You Are a Badass", "author": "Jen Sincero"},
        {"title": "The Happiness Advantage", "author": "Shawn Achor"},
        {"title": "The Four Agreements", "author": "Don Miguel Ruiz"},
        {"title": "Drive: The Surprising Truth About What Motivates Us", "author": "Daniel H. Pink"}
    ],
    "negative": [
        {"title": "Feeling Good: The New Mood Therapy", "author": "David D. Burns"},
        {"title": "The Subtle Art of Not Giving a F*ck", "author": "Mark Manson"},
        {"title": "Daring Greatly", "author": "Brené Brown"},
        {"title": "The Gifts of Imperfection", "author": "Brené Brown"},
        {"title": "Mindset: The New Psychology of Success", "author": "Carol S. Dweck"},
        {"title": "Radical Acceptance", "author": "Tara Brach"},
        {"title": "Lost Connections", "author": "Johann Hari"},
        {"title": "Emotional Agility", "author": "Susan David"},
        {"title": "Rising Strong", "author": "Brené Brown"},
        {"title": "When Things Fall Apart", "author": "Pema Chödrön"}
    ],
    "neutral": [
        {"title": "Sapiens: A Brief History of Humankind", "author": "Yuval Noah Harari"},
        {"title": "Educated", "author": "Tara Westover"},
        {"title": "Quiet: The Power of Introverts", "author": "Susan Cain"},
        {"title": "Thinking, Fast and Slow", "author": "Daniel Kahneman"},
        {"title": "The 7 Habits of Highly Effective People", "author": "Stephen R. Covey"},
        {"title": "Outliers: The Story of Success", "author": "Malcolm Gladwell"},
        {"title": "Grit: The Power of Passion and Perseverance", "author": "Angela Duckworth"},
        {"title": "Principles: Life and Work", "author": "Ray Dalio"},
        {"title": "Range: Why Generalists Triumph in a Specialized World", "author": "David Epstein"},
        {"title": "Deep Work: Rules for Focused Success in a Distracted World", "author": "Cal Newport"}
    ]
}

}

# Daily Affirmations
affirmations = [
    "💪 You are capable of achieving great things, one step at a time. Remember that even the tallest mountains are climbed by taking small, steady steps—keep going and you will reach your peak.",
    "🌠 As Walt Disney said, 'All our dreams can come true, if we have the courage to pursue them.' Let this remind you that your dreams are not impossible—they are waiting for you to act with courage, faith, and persistence.",
    "🔥 You are stronger than you know, and every challenge is a chance to grow. Obstacles are not meant to break you, but to reveal the depth of your resilience and the strength of your spirit.",
    "🌸 Like the lotus flower, you can rise above challenges and shine. No matter how muddy the water may be, you hold the power to bloom with grace, beauty, and determination.",
    "✨ Believe in yourself, for you have the power to shape your future. Every thought you nurture and every action you take creates ripples that build the life you truly desire.",
    "🔄 Every setback is a setup for an even greater comeback. Life’s detours may slow you down, but they also prepare you for opportunities far greater than what you had planned.",
    "🚀 Your potential is limitless, and every day is a new opportunity to discover it. The only limits are the ones you place on yourself—believe bigger, dream bolder, and act with confidence.",
    "🛤️ The journey may be tough, but so are you. With patience, persistence, and faith, you will find that every step forward, no matter how small, brings you closer to your destination.",
    "🌈 You deserve happiness, peace, and success—never doubt your worth. You are worthy of love, respect, and abundance simply because you exist, and nothing can take that away from you.",
    "⛰️ Each small step you take builds momentum for your big victories. Success doesn’t happen overnight—it is the result of consistent effort, persistence, and belief in your path.",
    "📖 You are not defined by your past, but by the choices you make today and tomorrow. Every sunrise gives you a fresh page to write a new story—make it one filled with courage and hope.",
    "☀️ Even on the hardest days, you are making progress simply by showing up. Strength is not about never struggling; it’s about showing up despite the struggle, again and again.",
    "🌍 The universe is full of opportunities waiting for you to claim them. Trust that life is unfolding in your favor, even when you cannot yet see the bigger picture.",
    "🧠 Your mind is powerful—feed it positivity, and your life will reflect it. Focus on thoughts of gratitude, hope, and strength, and watch how your reality transforms into something brighter.",
    "🪜 Challenges are not roadblocks; they are stepping stones toward greatness. Each difficulty you overcome prepares you for bigger opportunities and a stronger, wiser version of yourself.",
    "🌳 Remember: storms make trees take deeper roots. Just as the tree becomes stronger after enduring wind and rain, you too are growing deeper resilience with every challenge you face.",
    "⏳ Be patient with yourself; growth takes time, but every moment brings you closer to your dreams. Flowers do not bloom overnight, and neither does greatness—it unfolds beautifully, in its own time.",
    "❤️ You are enough, exactly as you are, and every day you are becoming more of who you are meant to be. You don’t need to prove your worth to anyone—your existence itself is proof of your significance.",
    "🎯 Success is not about speed, but consistency—every effort compounds into achievement. Small, repeated actions build habits, habits build character, and character shapes destiny.",
    "🌟 You carry light within you, and the world becomes brighter when you let it shine. Never dim yourself to fit in—your uniqueness is a gift that inspires and uplifts those around you."
]


def get_daily_affirmation():
    return random.choice(affirmations)

# Study Tips Based on Mood
study_tips = {
    "😊 Positive": [
        "You're in a great mood—use this energy to tackle a challenging topic with enthusiasm! When you’re motivated, your brain absorbs more information, so pick that subject you’ve been putting off and dive in with confidence.",
        "Try the Pomodoro technique: 25 minutes of focused study, then a 5-minute break to keep the good vibes going. During breaks, do something light and enjoyable like stretching, sipping water, or listening to your favorite upbeat song.",
        "Celebrate your progress by reviewing what you've learned today—it'll boost your confidence! Write down three key things you mastered, no matter how small, and remind yourself that consistency builds success.",
        "Channel your positivity into teaching someone else what you just studied. Explaining a concept aloud or writing it as if teaching can deepen your understanding and reinforce your memory.",
        "Use your high energy to plan ahead: organize your notes, create a study timetable, or map out difficult concepts you’ll conquer next. Preparation today makes tomorrow easier and more productive."
    ],
    "😞 Negative": [
        "Start with a small, manageable task to build momentum, as small wins can lift your spirits. For example, read one page, solve a single problem, or write a short summary. That spark of progress will push you forward.",
        "As Thomas Edison said, 'I have not failed. I've just found 10,000 ways that won't work.' Remember that mistakes are not signs of weakness but stepping stones to mastery. Each attempt builds resilience and insight.",
        "Study in a comfortable space with some light background music to ease your mind. Create an environment that feels calm, with minimal distractions, maybe add a cup of tea or water nearby to refresh yourself.",
        "Practice self-kindness: remind yourself that it’s okay to have off days. Instead of pushing too hard, choose gentle learning methods like watching an explainer video, using flashcards, or revising notes in smaller portions.",
        "Visualize your end goal—a completed degree, a rewarding career, or simply understanding this one tough topic. Motivation often comes when we reconnect with the bigger picture of why we started studying in the first place."
    ],
    "😐 Neutral": [
        "Break your study session into smaller chunks to maintain focus and avoid overwhelm. Aim for 30–40 minutes of active learning followed by a meaningful 10-minute break where you relax or stretch.",
        "Try summarizing key points in your own words to make studying more engaging. Creating mind maps, flowcharts, or simple doodles can make abstract ideas more concrete and easier to recall later.",
        "Set a clear goal for this session, like mastering one concept, to stay on track. Clarity fuels productivity—write your goal on a sticky note and place it where you can see it as a reminder.",
        "Use active recall by closing your notes and testing yourself on what you just learned. This strengthens memory and helps you identify knowledge gaps you might miss with passive rereading.",
        "Switch up your study method to spark engagement—if you usually read, try recording yourself explaining the topic, or quiz yourself with flashcards. A slight change in routine can reignite focus."
    ]
}


def get_study_tips(mood):
    return random.sample(study_tips[mood], min(2, len(study_tips[mood])))

# Breathing Exercise
def breathing_exercise(sid):
    socketio.emit('ai_response', "Anchor: Let's do a quick breathing exercise to help you relax.", room=sid)
    socketio.emit('ai_response', "Inhale deeply for 4 seconds, hold for 4 seconds, exhale for 4 seconds. Repeat 3 times.", room=sid)
    socketio.emit('ai_response', "1. Inhale... Hold... Exhale...", room=sid)
    socketio.sleep(12)
    socketio.emit('ai_response', "2. Inhale... Hold... Exhale...", room=sid)
    socketio.sleep(12)
    socketio.emit('ai_response', "3. Inhale... Hold... Exhale...", room=sid)
    socketio.sleep(12)
    socketio.emit('ai_response', "Anchor: Great job! You should feel a bit calmer now. Ready to continue?", room=sid)

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

# Sentiment Analysis & Mood Logging
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
    timestamps, scores = [], []
    if not os.path.exists("user.txt"):
        return "<p>Anchor: No mood data found yet.</p>"
    with open("user.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 3)
            try:
                if len(parts) >= 2:
                    timestamps.append(parts[0])
                    scores.append(float(parts[1]))
            except ValueError:
                continue
    if not scores:
        return "<p>Anchor: No mood data found yet.</p>"
    plt.style.use('dark_background')  # Set dark theme
    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, scores, marker="o", linestyle="-", color="cyan", label="Sentiment Score")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.gca().get_xaxis().set_visible(False)  # Remove x-axis
    plt.ylabel("Sentiment (−1 = Negative, +1 = Positive)")
    plt.title("Mood Tracker Sentiment Trend")
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f'<img src="data:image/png;base64,{img_base64}" alt="Mood Analysis Plot" style="max-width:100%;">'

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
        output += "<h3>🧘 Meditative Music:</h3><ul>"
        for i, music in enumerate(recs["meditative"], 1):
            output += f"<li>{i}. {music['title']} - <a href='{music['url']}' target='_blank'>{music['url']}</a></li>"
        output += "</ul>"
    if "movies" in recs and recs["movies"]:
        output += "<h3>🎬 Movies:</h3><ul>"
        for i, movie in enumerate(recs["movies"], 1):
            output += f"<li>{i}. {movie['title']} - <a href='{movie['youtube_link']}' target='_blank'>{movie['youtube_link']}</a><br>{movie['description']}</li>"
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
    if 'therapy_responses' not in session:
        session['therapy_responses'] = []
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
    emit('ai_response', f"Anchor: Here's your daily affirmation: {get_daily_affirmation()}")
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

# Add therapy questions and state management
therapy_questions = [
    "How have you been feeling emotionally lately? Are there any specific emotions standing out for you? 😊",
    "What’s been the biggest source of stress or challenge in your life right now? 😔",
    "How has your sleep been? Are you getting enough rest, or is something keeping you up? 😴",
    "How are your relationships with friends, family, or others? Anything you’d like to share? 🤝",
    "What do you do to cope when things feel overwhelming? Are there strategies that help you feel better? 🛠️",
    "Is there anything in your lifestyle, like diet or exercise, that you feel impacts your mental well-being? 🥗",
    "Looking forward, what’s one thing you’d like to work on to feel more balanced or supported? 🌟"
]

# Recommendations for post-therapy guidance
therapy_recommendations = {
    "stress": [
        "Try a 5-minute mindfulness meditation daily to reduce stress. Apps like Calm or Headspace can guide you. 🧘",
        "Journaling your thoughts for 10 minutes each evening can help process stress. Write without judgment—what’s on your mind? 📝"
    ],
    "sleep": [
        "Establish a bedtime routine: avoid screens 30 minutes before bed and try reading or light stretching. 😴",
        "Consider a short guided sleep meditation to ease into rest. YouTube has free options like 'Sleep Meditation for Beginners.' 🛌"
    ],
    "relationships": [
        "Open communication is key. Try scheduling a heart-to-heart with someone you trust to share how you’re feeling. 🤝",
        "Set boundaries to protect your emotional energy. It’s okay to say ‘no’ when you need space. 🛑"
    ],
    "general": [
        "Practice self-compassion: remind yourself it’s okay to have tough days. You’re doing your best. ❤️",
        "Consider speaking with a licensed therapist for professional support. Platforms like BetterHelp can connect you. 🩺"
    ]
}

def format_therapy_summary(responses):
    summary = "<p>Anchor: Thank you for sharing so openly. Here’s a summary of our session:</p><ul>"
    for i, (question, response) in enumerate(responses, 1):
        summary += f"<li>{i}. <b>{question}</b><br>Your response: {response}</li>"
    summary += "</ul><p>Based on what you shared, here are some tailored suggestions:</p><ul>"
    
    # Analyze responses for key themes
    stress_mentioned = any("stress" in resp.lower() or "pressure" in resp.lower() for _, resp in responses)
    sleep_mentioned = any("sleep" in resp.lower() or "tired" in resp.lower() for _, resp in responses)
    relationship_mentioned = any("friend" in resp.lower() or "family" in resp.lower() or "relationship" in resp.lower() for _, resp in responses)
    
    # Provide recommendations based on themes
    if stress_mentioned:
        for rec in therapy_recommendations["stress"]:
            summary += f"<li>{rec}</li>"
    if sleep_mentioned:
        for rec in therapy_recommendations["sleep"]:
            summary += f"<li>{rec}</li>"
    if relationship_mentioned:
        for rec in therapy_recommendations["relationships"]:
            summary += f"<li>{rec}</li>"
    for rec in therapy_recommendations["general"]:  # Always include general recommendations
        summary += f"<li>{rec}</li>"
    
    summary += "</ul><p>Anchor: I’m here whenever you need to talk again. Keep shining! ✨</p>"
    return summary

@socketio.on('user_message')
def handle_user_message(msg):
    sid = request.sid
    if msg.lower() == "exit":
        emit('ai_response', "Chat ended. Take care!")
        return

    now, mood, score, follow_up = log_mood(msg)
    session['last_mood'] = mood

    state = session.get('state')

    # Handle Therapy Mode
    if msg.lower() == "therapy start":
        session['state'] = 'therapy_1'
        session['therapy_responses'] = []
        emit('ai_response', "Anchor: I’m here to listen and support you in therapy mode. Let’s begin with a safe, open conversation.")
        emit('ai_response', therapy_questions[0])
        return
    elif msg.lower() == "stop therapy" and state and state.startswith('therapy_'):
        # Summarize and provide recommendations
        summary = format_therapy_summary(session['therapy_responses'])
        session['state'] = None
        session['therapy_responses'] = []
        emit('ai_response', summary)
        emit('ai_response', "Anchor: We’ve exited therapy mode. I’m here for you—how can I support you now? 😊")
        return
    elif state and state.startswith('therapy_'):
        # Store response and proceed to next question
        question_index = int(state.split('_')[1]) - 1
        session['therapy_responses'].append((therapy_questions[question_index], msg))
        
        # Move to next question or end session
        next_index = question_index + 1
        if next_index < len(therapy_questions):
            session['state'] = f'therapy_{next_index + 1}'
            emit('ai_response', therapy_questions[next_index])
        else:
            # End of therapy session
            summary = format_therapy_summary(session['therapy_responses'])
            session['state'] = None
            session['therapy_responses'] = []
            emit('ai_response', summary)
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
                completion = client.chat.completions.create(
                    model="google/gemma-2-9b-it",
                    messages=session['messages']
                )
                response_text = completion.choices[0].message.get("content", "")
                emit('ai_response', f"Anchor: {response_text.strip()}")
                session['messages'].append({"role": "assistant", "content": response_text})
            except Exception as e:
                emit('ai_response', f"Anchor: I found those search results for you! Is there anything specific you'd like to know about them?")
        return

    if "daily affirmation" in msg.lower():
        emit('ai_response', f"Anchor: Here's your affirmation: {get_daily_affirmation()}")
        return

    if "study tips" in msg.lower() or "help me study" in msg.lower():
        tips = get_study_tips(mood)
        tip_text = "<p>Anchor: Here are some study tips tailored to your mood:</p><ul>"
        for i, tip in enumerate(tips, 1):
            tip_text += f"<li>{i}. {tip}</li>"
        tip_text += "</ul>"
        emit('ai_response', tip_text)
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

    # Add proactive mood/life check after processing commands, but not during state flows
    if not state and random.random() < 0.3:  # Reduced frequency to 30%
        mood_check = random.choice([
            "I'm here for you—how's your heart feeling today?",
            "What's been going on in your world lately?",
            "How can I support you today? Feeling up or down?",
            "Tell me, what's been the highlight of your day so far?"
        ])
        emit('ai_response', f"Anchor: {mood_check}")
        session['messages'].append({"role": "assistant", "content": mood_check})

    # Process with AI if no specific command matched
    session['messages'].append({"role": "user", "content": msg})
    if len(session['messages']) > 20:
        session['messages'] = session['messages'][-20:]

    try:
        completion = client.chat.completions.create(
            model="google/gemma-2-9b-it",
            messages=session['messages']
        )
        response_text = completion.choices[0].message.get("content", "").strip()

        # If AI doesn't know something, try web search
        if any(phrase in response_text.lower() for phrase in ["i don't know", "i'm not sure", "i cannot find"]):
            search_results = search_and_fetch_content(msg)
            if search_results:
                combined_content = "\n\n".join([r["content"][:1000] for r in search_results])  # Limit content
                session['messages'].append({"role": "system", "content": f"Use the following web content to answer the user:\n{combined_content}"})
                completion = client.chat.completions.create(
                    model="google/gemma-2-9b-it",
                    messages=session['messages']
                )
                response_text = completion.choices[0].message.get("content", "").strip()

        emit('ai_response', f"Anchor: {response_text}")
        session['messages'].append({"role": "assistant", "content": response_text})

    except Exception as e:
        emit('ai_response', f"Anchor: {follow_up}")

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
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)