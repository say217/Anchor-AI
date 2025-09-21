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
import numpy as np
import time
import eventlet
# Initialize Hugging Face client
HF_TOKEN = os.getenv('HF_TOKEN')
client = InferenceClient(token=HF_TOKEN)

# Global task and goal lists - persist to /tmp
tasks = []
goals = []

data_dir = os.getenv('DATA_DIR', '/tmp')  # Use /data on Fly.io

def load_data():
    global tasks, goals
    try:
        with open(os.path.join(data_dir, "tasks.json"), "r") as f:
            tasks = json.load(f)
            tasks = [(task[0], datetime.fromisoformat(task[1]), task[2]) for task in tasks]
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        tasks = []
    try:
        with open(os.path.join(data_dir, "goals.json"), "r") as f:
            goals = json.load(f)
            goals = [(goal[0], datetime.fromisoformat(goal[1]), goal[2]) for goal in goals]
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        goals = []

def save_tasks():
    tasks_to_save = [(task[0], task[1].isoformat(), task[2]) for task in tasks]
    with open(os.path.join(data_dir, "tasks.json"), "w") as f:
        json.dump(tasks_to_save, f)

def save_goals():
    goals_to_save = [(goal[0], goal[1].isoformat(), goal[2]) for goal in goals]
    with open(os.path.join(data_dir, "goals.json"), "w") as f:
        json.dump(goals_to_save, f)

# Initial messages for AI
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

# Predefined recommendation lists (unchanged from original)
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
        "url": "https://www.youtube.com/watch?v=SP0rC1J6EJg",
        "description": "Tips and speeches to maintain laser-like focus. Perfect for students, athletes, and professionals who need to eliminate distractions."
    },
    {
        "title": "BOUNCE BACK - Resilience Motivation",
        "url": "https://www.youtube.com/watch?v=eoMYylO7u9g",
        "description": "Learn to recover from setbacks stronger than before. Reminds you that failure is not final—it's only a stepping stone to success."
    },
    {
        "title": "EMBRACE CHANGE - Adaptability Inspiration",
        "url": "https://www.youtube.com/watch?v=pUmTQ-86-YI",
        "description": "Motivation to thrive in times of change and uncertainty. Teaches flexibility, adaptability, and the courage to step outside your comfort zone."
    },

    
    {
        "title": "Your Elusive Creative Genius | Elizabeth Gilbert (TED)",
        "url": "https://www.youtube.com/watch?v=86x-u-tz0MA",
        "description": "A powerful TED Talk where Elizabeth Gilbert discusses creativity, fear of failure, and how to keep going despite challenges."
    },
    {
        "title": "The Power of Vulnerability | Brené Brown (TED)",
        "url": "https://www.youtube.com/watch?v=iCvmsMzlF7o",
        "description": "Brené Brown explores courage, vulnerability, and connection, showing how embracing vulnerability leads to strength and authenticity."
    },
    {
        "title": "How Great Leaders Inspire Action | Simon Sinek (TED)",
        "url": "https://www.youtube.com/watch?v=qp0HIF3SfI4",
        "description": "Simon Sinek introduces the 'Golden Circle' and explains how great leaders inspire action through purpose and vision."
    },
    {
        "title": "The Puzzle of Motivation | Dan Pink (TED)",
        "url": "https://www.youtube.com/watch?v=rrkrvAUbU9Y",
        "description": "Dan Pink reveals surprising truths about motivation, emphasizing autonomy, mastery, and purpose over traditional rewards."
    },
    {
        "title": "Your Body Language Shapes Who You Are | Amy Cuddy (TED)",
        "url": "https://www.youtube.com/watch?v=Ks-_Mh1QhMc",
        "description": "Amy Cuddy shares how 'power posing' can influence confidence and success, showing how body language shapes self-belief."
    },
    {
        "title": "Grit: The Power of Passion and Perseverance | Angela Lee Duckworth (TED)",
        "url": "https://www.youtube.com/watch?v=H14bBuluwB8",
        "description": "Angela Duckworth explains why grit—passion and perseverance—is more important than talent in achieving long-term success."
    },
         
     {
        "title": "The Ed Mylett Show Podcast | Mindset & Motivation",
        "url": "https://www.youtube.com/watch?v=oFf9FAuv0WA&list=PLt590l0kppMm6JxKMSkEkX4SQ3zPSOkId",
        "description": "Ed Mylett interviews top performers in business, sports, and life to uncover the secrets of success and peak performance."
    },
    {
        "title": "Impact Theory with Tom Bilyeu",
        "url": "https://www.youtube.com/watch?v=ziLmtuLm-LU",
        "description": "Tom Bilyeu hosts impactful conversations with world-class thinkers, entrepreneurs, and leaders sharing success principles."
    },
    {
        "title": "The School of Greatness Podcast | Lewis Howes",
        "url": "https://www.youtube.com/watch?v=3ezAOgZGXKw",
        "description": "Lewis Howes shares stories and strategies from inspirational leaders and world-class achievers."
    },
    {
        "title": "The Mindset Mentor Podcast | Rob Dial",
        "url": "https://www.youtube.com/watch?v=4kIDyv39Hlk",
        "description": "Bite-sized daily motivational podcasts that help shift mindset and achieve success."
    },
    {
        "title": "The Tony Robbins Podcast",
        "url": "https://www.youtube.com/watch?v=BwjnG45zO5U",
        "description": "Tony Robbins shares strategies, interviews, and lessons on achieving personal and professional breakthroughs."
    },
    {
        "title": "Oprah’s SuperSoul Conversations",
        "url": "https://www.youtube.com/watch?v=fBWStmXMnUM",
        "description": "Oprah Winfrey shares deep conversations with thought leaders, helping you awaken to your best self."
    },
    {
        "title": "Rich Roll Podcast | Motivation for Growth",
        "url": "https://www.youtube.com/watch?v=jwZ-C_3tMBU",
        "description": "Rich Roll shares inspiring stories of transformation, health, and resilience."
    },
    {
        "title": "The Tim Ferriss Show | Productivity & Success",
        "url": "https://www.youtube.com/watch?v=Kd06uvinqLI",
        "description": "Tim Ferriss interviews top performers to uncover habits and tools for success."
    },
    {
        "title": "David Goggins Biography | Stay Hard",
        "url": "https://www.youtube.com/watch?v=dIM7E8e9JKY",
        "description": "Life story of David Goggins, Navy SEAL and ultramarathon runner, proving how mental toughness beats all odds."
    },
    {
        "title": "Elon Musk Biography | Innovator & Visionary",
        "url": "https://www.youtube.com/watch?v=BfsuFXpW5Ns",
        "description": "The inspiring journey of Elon Musk, from startups to SpaceX and Tesla, showing the power of ambition and resilience."
    },
    {
        "title": "Steve Jobs Biography | Think Different",
        "url": "https://www.youtube.com/watch?v=s4pVFLUlx8g",
        "description": "Steve Jobs’ journey at Apple and Pixar, and how his vision reshaped technology and creativity."
    },
    {
        "title": "Muhammad Ali Biography | The Greatest",
        "url": "https://www.youtube.com/watch?v=X-NW3NlL7W0",
        "description": "The life and struggles of Muhammad Ali, showcasing his fight both inside and outside the boxing ring."
    },
    {
        "title": "Nelson Mandela Biography | Freedom Fighter",
        "url": "https://www.youtube.com/watch?v=PyfOrbO0xf4",
        "description": "The remarkable life of Nelson Mandela, a symbol of peace, resilience, and justice."
    },
    {
        "title": "Kobe Bryant Biography | Mamba Mentality",
        "url": "https://www.youtube.com/watch?v=GE0UAdxPTc0",
        "description": "A tribute to Kobe Bryant’s relentless dedication, passion, and pursuit of greatness."
    },
    {
        "title": "Barack Obama Biography | Yes We Can",
        "url": "https://www.youtube.com/watch?v=Fe751kMBwms",
        "description": "Journey of Barack Obama from community leader to President of the USA, inspiring millions with hope and change."
    },
    {
        "title": "Arnold Schwarzenegger Biography | From Bodybuilder to Icon",
        "url": "https://www.youtube.com/watch?v=kxHygJLwmnk",
        "description": "The journey of Arnold Schwarzenegger from bodybuilding champion to Hollywood star and governor."
    },
    {
        "title": "J.K. Rowling Biography | From Failure to Harry Potter",
        "url": "https://www.youtube.com/watch?v=L2rR5RuJEPc",
        "description": "The inspiring story of J.K. Rowling, who turned rejection and struggle into global success."
    },
    {
        "title": "Albert Einstein Biography | Genius of Physics",
        "url": "https://www.youtube.com/watch?v=co3FrMo4WXc",
        "description": "Life of Albert Einstein, exploring his genius, discoveries, and struggles."
    },
    {
        "title": "Netaji Subhas Chandra Bose | Father of a Nation",
        "url": "https://www.youtube.com/watch?v=96oyCiAEb4M",
        "description": "The inspiring life of Mahatma Gandhi, who led India’s independence movement through peace and non-violence."
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
# Daily Affirmations (shortened for brevity)
# Daily Affirmations
affirmations = [
    "💪 You are capable of achieving great things, one step at a time. Remember that even the tallest mountains are climbed by taking small, steady steps—keep going and you will reach your peak.",
    "🌠 As Walt Disney said, 'All our dreams can come true, if we have the courage to pursue them.' Let this remind you that your dreams are not impossible—they are waiting for you to act with courage, faith, and persistence.",
    "🔥 You are stronger than you know, and every challenge is a chance to grow. Obstacles are not meant to break you, but to reveal the depth of your resilience and the strength of your spirit.",
    "🌸 Like the lotus flower, you can rise above challenges and shine. No matter how muddy the water may be, you hold the power to bloom with grace, beauty, and determination.",
    "✨ Believe in yourself, for you have the power to shape your future. Every thought you nurture and every action you take creates ripples that build the life you truly desire.",
    "🔄 Every setback is a setup for an even greater comeback. Life's detours may slow you down, but they also prepare you for opportunities far greater than what you had planned.",
    "🚀 Your potential is limitless, and every day is a new opportunity to discover it. The only limits are the ones you place on yourself—believe bigger, dream bolder, and act with confidence.",
    "🛤️ The journey may be tough, but so are you. With patience, persistence, and faith, you will find that every step forward, no matter how small, brings you closer to your destination.",
    "🌈 You deserve happiness, peace, and success—never doubt your worth. You are worthy of love, respect, and abundance simply because you exist, and nothing can take that away from you.",
    "⛰️ Each small step you take builds momentum for your big victories. Success doesn't happen overnight—it is the result of consistent effort, persistence, and belief in your path.",
    "📖 You are not defined by your past, but by the choices you make today and tomorrow. Every sunrise gives you a fresh page to write a new story—make it one filled with courage and hope.",
    "☀️ Even on the hardest days, you are making progress simply by showing up. Strength is not about never struggling; it's about showing up despite the struggle, again and again.",
    "🌍 The universe is full of opportunities waiting for you to claim them. Trust that life is unfolding in your favor, even when you cannot yet see the bigger picture.",
    "🧠 Your mind is powerful—feed it positivity, and your life will reflect it. Focus on thoughts of gratitude, hope, and strength, and watch how your reality transforms into something brighter.",
    "🪜 Challenges are not roadblocks; they are stepping stones toward greatness. Each difficulty you overcome prepares you for bigger opportunities and a stronger, wiser version of yourself.",
    "🌳 Remember: storms make trees take deeper roots. Just as the tree becomes stronger after enduring wind and rain, you too are growing deeper resilience with every challenge you face.",
    "⏳ Be patient with yourself; growth takes time, but every moment brings you closer to your dreams. Flowers do not bloom overnight, and neither does greatness—it unfolds beautifully, in its own time.",
    "❤️ You are enough, exactly as you are, and every day you are becoming more of who you are meant to be. You don't need to prove your worth to anyone—your existence itself is proof of your significance.",
    "🎯 Success is not about speed, but consistency—every effort compounds into achievement. Small, repeated actions build habits, habits build character, and character shapes destiny.",
    "🌟 You carry light within you, and the world becomes brighter when you let it shine. Never dim yourself to fit in—your uniqueness is a gift that inspires and uplifts those around you."
]


def get_daily_affirmation():
    return random.choice(affirmations)


# Study Tips Based on Mood
study_tips = {
    "😊 Positive": [
        "You're in a great mood—use this energy to tackle a challenging topic with enthusiasm! When you're motivated, your brain absorbs more information, so pick that subject you've been putting off and dive in with confidence.",
        "Try the Pomodoro technique: 25 minutes of focused study, then a 5-minute break to keep the good vibes going. During breaks, do something light and enjoyable like stretching, sipping water, or listening to your favorite upbeat song.",
        "Celebrate your progress by reviewing what you've learned today—it'll boost your confidence! Write down three key things you mastered, no matter how small, and remind yourself that consistency builds success.",
        "Channel your positivity into teaching someone else what you just studied. Explaining a concept aloud or writing it as if teaching can deepen your understanding and reinforce your memory.",
        "Use your high energy to plan ahead: organize your notes, create a study timetable, or map out difficult concepts you'll conquer next. Preparation today makes tomorrow easier and more productive."
    ],
    "😞 Negative": [
        "Start with a small, manageable task to build momentum, as small wins can lift your spirits. For example, read one page, solve a single problem, or write a short summary. That spark of progress will push you forward.",
        "As Thomas Edison said, 'I have not failed. I've just found 10,000 ways that won't work.' Remember that mistakes are not signs of weakness but stepping stones to mastery. Each attempt builds resilience and insight.",
        "Study in a comfortable space with some light background music to ease your mind. Create an environment that feels calm, with minimal distractions, maybe add a cup of tea or water nearby to refresh yourself.",
        "Practice self-kindness: remind yourself that it's okay to have off days. Instead of pushing too hard, choose gentle learning methods like watching an explainer video, using flashcards, or revising notes in smaller portions.",
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
    socketio.emit('ai_response', "Anchor: Let's do a quick breathing exercise.", room=sid)
    socketio.emit('ai_response', "Inhale for 4 seconds, hold for 4, exhale for 4. Repeat 3 times.", room=sid)
    socketio.emit('ai_response', "1. Inhale... Hold... Exhale...", room=sid)
    socketio.sleep(12)
    socketio.emit('ai_response', "2. Inhale... Hold... Exhale...", room=sid)
    socketio.sleep(12)
    socketio.emit('ai_response', "3. Inhale... Hold... Exhale...", room=sid)
    socketio.sleep(12)
    socketio.emit('ai_response', "Anchor: Great job! Feel calmer? Ready to continue?", room=sid)

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
            goals.pop(0)
            socketio.emit('ai_response', "Anchor: Goal limit reached. Removed oldest goal.")
        goal_id = str(int(time.time() * 1000))
        goals.append((goal_name, goal_datetime, goal_id))
        save_goals()
        socketio.emit('update_goals', format_goals())
        return f"Anchor: Goal '{goal_name}' set for {goal_datetime.strftime('%Y-%m-%d')}."
    except ValueError:
        return "Anchor: Invalid date format. Use YYYY-MM-DD."

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
            "You seem in a great mood! What's got you smiling?",
            "Love your positive vibes! What's going well?"
        ])
    elif compound_score <= -0.05:
        mood = "😞 Negative"
        follow_up = random.choice([
            "I'm here for you. Want to share what's tough?",
            "Sounds like you're feeling down. Can I help?"
        ])
    else:
        mood = "😐 Neutral"
        follow_up = random.choice([
            "You seem steady. What's on your mind?",
            "Everything okay? Tell me what's up!"
        ])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{now},{compound_score},{mood},{user_text}\n"
    with open(os.path.join(data_dir, "user.txt"), "a", encoding="utf-8") as f:
        f.write(entry)
    return now, mood, compound_score, follow_up

def get_mood_plot():
    timestamps, scores, moods = [], [], []
    if not os.path.exists(os.path.join(data_dir, "user.txt")):
        return "<div class='mood-plot-container'><p>Anchor: No mood data found yet. Start chatting to track your emotions!</p></div>"
    
    try:
        with open(os.path.join(data_dir, "user.txt"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 3)
                if len(parts) >= 3:
                    try:
                        timestamp = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
                        timestamps.append(timestamp)
                        scores.append(float(parts[1]))
                        moods.append(parts[2])
                    except ValueError:
                        continue
    except Exception as e:
        return f"<div class='mood-plot-container'><p>Anchor: Error reading mood data: {str(e)}</p></div>"
    
    if not scores:
        return "<div class='mood-plot-container'><p>Anchor: No valid mood data yet. Keep chatting!</p></div>"

    data_count = len(scores)
    figsize = (8, 4) if data_count <= 20 else (10, 5)
    marker_size = 6 if data_count <= 20 else 4
    line_width = 2 if data_count <= 20 else 1.5
    title_size = 12 if data_count <= 20 else 14

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    colors = ['#4CAF50' if "Positive" in mood else '#F44336' if "Negative" in mood else '#FFC107' for mood in moods]
    ax.scatter(range(len(scores)), scores, c=colors, s=marker_size*10, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.plot(range(len(scores)), scores, color='cyan', alpha=0.7, linewidth=line_width)
    
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0.3, color="green", linestyle=":", linewidth=0.8, alpha=0.4, label="Positive Threshold")
    ax.axhline(-0.3, color="red", linestyle=":", linewidth=0.8, alpha=0.4, label="Negative Threshold")
    
    ax.set_ylabel("Sentiment Score", color='white', fontsize=10)
    ax.set_xlabel("Conversation Timeline", color='white', fontsize=10)
    ax.set_title("Your Emotional Journey", color='white', fontsize=title_size, pad=20)
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(True, alpha=0.2)
    
    if data_count <= 20:
        for i, (score, mood) in enumerate(zip(scores, moods)):
            if abs(score) > 0.5:
                emoji = "😊" if "Positive" in mood else "😞" if "Negative" in mood else "😐"
                ax.annotate(emoji, (i, score), xytext=(0, 10), textcoords='offset points', 
                           ha='center', fontsize=8, alpha=0.8)
    
    if data_count > 10:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a1a')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    avg_score = np.mean(scores)
    recent_trend = "improving" if len(scores) >= 3 and scores[-1] > scores[-3] else "stable" if len(scores) >= 3 and abs(scores[-1] - scores[-3]) < 0.1 else "declining"
    positive_count = sum(1 for s in scores if s > 0.05)
    negative_count = sum(1 for s in scores if s < -0.05)
    neutral_count = len(scores) - positive_count - negative_count
    
    analysis_html = f"""
    <div class='mood-plot-container' style='background: #1a1a1a; padding: 20px; border-radius: 10px;'>
        <img src="data:image/png;base64,{img_base64}" alt="Mood Analysis Plot" style="max-width:100%; border-radius: 8px;">
        <div class='mood-stats' style='color: white;'>
            <h3 style='color: cyan;'>Your Emotional Insights</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>
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
            <div style='background: #000; padding: 15px; border-radius: 5px;'>
                <strong>Overall Mood Trend:</strong> <span style='color: {"#4CAF50" if avg_score > 0.05 else "#F44336" if avg_score < -0.05 else "#FFC107"};'>{recent_trend.title()}</span>
                <br>
                <strong>Average Sentiment:</strong> <span style='color: {"#4CAF50" if avg_score > 0 else "#F44336" if avg_score < 0 else "#FFC107"};'>{avg_score:.3f}</span>
            </div>
            <p style='font-size: 14px; color: #ccc;'>
                💙 Your emotions matter, and I'm here for all of them!
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
        return [item['link'] for item in response.get('items', [])]
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
        result = {"title": title, "url": url, "description": preview['desc']}
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

# Book Recommendations
def get_book_summary(book_title):
    query = f"{book_title} book summary"
    search_results = search_and_fetch_content(query, num_results=1)
    if search_results:
        content = search_results[0]["content"]
        return content[:300].strip() + "..." if len(content) > 300 else content
    return "Summary not available."

def get_book_recommendations(mood):
    mood_key = mood.split()[1].lower()
    books = random.sample(recommendations["books"].get(mood_key, []), min(3, len(recommendations["books"].get(mood_key, []))))
    return [
        {
            "title": book["title"],
            "author": book["author"],
            "suggestion": f"This book by {book['author']} helps with motivation and growth.",
            "summary": get_book_summary(book["title"])
        }
        for book in books
    ]

def format_book_recommendations(books):
    output = "<p>Anchor: Here are some book suggestions:</p><ul>"
    for book in books:
        output += f"<li><b>{book['title']}</b> by {book['author']}<br>{book['suggestion']}<br>Summary: {book['summary']}</li>"
    output += "</ul>"
    return output

# Other Recommendations
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
    if "songs" in recs:
        output += "<h3>🎶 Bollywood Songs:</h3><ul>"
        for i, song in enumerate(recs["songs"], 1):
            output += f"<li>{i}. {song['title']} by {song['singer']} - <a href='{song['youtube_link']}' target='_blank'>{song['youtube_link']}</a></li>"
        output += "</ul>"
    if "videos" in recs:
        output += "<h3>📽️ Motivational Videos:</h3><ul>"
        for i, video in enumerate(recs["videos"], 1):
            output += f"<li>{i}. {video['title']} - <a href='{video['url']}' target='_blank'>{video['url']}</a></li>"
        output += "</ul>"
    if "meditative" in recs:
        output += "<h3>Meditative Music:</h3><ul>"
        for i, music in enumerate(recs["meditative"], 1):
            output += f"<li>{i}. {music['title']} - <a href='{music['url']}' target='_blank'>{music['url']}</a></li>"
        output += "</ul>"
    if "movies" in recs:
        output += "<h3>Movies:</h3><ul>"
        for i, movie in enumerate(recs["movies"], 1):
            output += f"<li>{i}. {movie['title']} - <a href='{movie['youtube_link']}' target='_blank'>{movie['youtube_link']}</a><br>{movie['description']}</li>"
        output += "</ul>"
    return output

# Scheduler Functions
def add_task(task_name, task_time, task_date):
    try:
        task_datetime = datetime.strptime(f"{task_date} {task_time}", "%Y-%m-%d %H:%M")
        if len(tasks) >= 10:
            tasks.pop(0)
            socketio.emit('ai_response', "Anchor: Task limit reached. Removed oldest task.")
        task_id = str(int(time.time() * 1000))
        tasks.append((task_name, task_datetime, task_id))
        save_tasks()
        socketio.emit('update_tasks', format_tasks())
        return f"Task '{task_name}' scheduled for {task_datetime.strftime('%Y-%m-%d %H:%M')}"
    except ValueError:
        return "Anchor: Invalid date/time format. Use YYYY-MM-DD for date and HH:MM for time."

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

# Flask App
app = Flask(__name__)
app.secret_key = 'super_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

load_data()

# Background checker
def background_checker():
    while True:
        cron_check_tasks()
        cron_check_goals()
        cron_clear_mood()
        time.sleep(60)  # Check every minute

socketio.start_background_task(background_checker)

def cron_check_tasks():
    now = datetime.now()
    tasks_to_remove = []
    for task_name, task_time, task_id in tasks[:]:
        time_diff = (task_time - now).total_seconds()
        if time_diff <= 0:
            socketio.emit('popup_notification', {
                'type': 'task', 'title': 'Task Reminder',
                'message': f"It's time for: {task_name}", 'icon': 'fas fa-tasks'
            })
            tasks_to_remove.append((task_name, task_time, task_id))
        elif time_diff <= 3600:
            socketio.emit('popup_notification', {
                'type': 'task', 'title': 'Task Reminder',
                'message': f"Reminder: '{task_name}' is due in less than an hour!",
                'icon': 'fas fa-clock'
            })
    for task in tasks_to_remove:
        tasks.remove(task)
    save_tasks()

def cron_check_goals():
    now = datetime.now()
    goals_to_remove = []
    for goal_name, goal_time, goal_id in goals[:]:
        time_diff = (goal_time - now).total_seconds()
        if time_diff <= 0:
            socketio.emit('popup_notification', {
                'type': 'goal', 'title': 'Goal Deadline',
                'message': f"Deadline reached for goal: {goal_name}",
                'icon': 'fas fa-flag-checkered'
            })
            goals_to_remove.append((goal_name, goal_time, goal_id))
        elif time_diff <= 24 * 3600:
            socketio.emit('popup_notification', {
                'type': 'goal', 'title': 'Goal Reminder',
                'message': f"Reminder: Goal '{goal_name}' is due tomorrow!",
                'icon': 'fas fa-exclamation-triangle'
            })
    for goal in goals_to_remove:
        goals.remove(goal)
    save_goals()

def cron_clear_mood():
    if os.path.exists(os.path.join(data_dir, "user.txt")):
        try:
            with open(os.path.join(data_dir, "user.txt"), "r", encoding="utf-8") as f:
                lines = f.readlines()
            now = datetime.now()
            valid_lines = [
                line for line in lines
                if (now - datetime.strptime(line.split(",", 1)[0], "%Y-%m-%d %H:%M:%S")).total_seconds() <= 48 * 3600
            ]
            with open(os.path.join(data_dir, "user.txt"), "w", encoding="utf-8") as f:
                f.writelines(valid_lines)
        except Exception:
            pass

# Preview Function
def get_preview(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
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

@app.route('/cron/check_tasks')
def cron_check_tasks_route():
    cron_check_tasks()
    return jsonify({'status': 'success'})

@app.route('/cron/check_goals')
def cron_check_goals_route():
    cron_check_goals()
    return jsonify({'status': 'success'})

@app.route('/cron/clear_mood')
def cron_clear_mood_route():
    cron_clear_mood()
    return jsonify({'status': 'success'})

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

@socketio.on('user_message')
def handle_user_message(msg):
    sid = request.sid
    if msg.lower() == "exit":
        emit('ai_response', "Chat ended. Take care!")
        return

    now, mood, score, follow_up = log_mood(msg)
    session['last_mood'] = mood
    state = session.get('state')

    if msg.lower() in ["therapy start", "start therapy", "therapy mode"]:
        session['state'] = 'therapy_active'
        session['therapy_start_time'] = datetime.now()
        session['messages'].append({
            "role": "system",
            "content": "User activated therapy mode. Provide empathetic, therapeutic responses."
        })
        emit('ai_response', "Anchor: I'm here to listen with my whole heart. What's been on your mind? 💙")
        return
    elif msg.lower() in ["stop therapy", "end therapy", "exit therapy"] and state == 'therapy_active':
        session['state'] = None
        emit('ai_response', "Anchor: Thank you for sharing. I'm proud of your courage. I'm always here. 💙")
        emit('ai_response', "Anchor: We've exited therapy mode. How can I support you now?")
        return

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
        with open(os.path.join(data_dir, "gratitude.txt"), "a", encoding="utf-8") as f:
            f.write(f"{now},{session['temp_gratitude1']},{session['temp_gratitude2']},{gratitude3}\n")
        emit('ai_response', "Anchor: Beautiful reflections! Keep shining!")
        session['state'] = None
        return

    if "show my mood analysis" in msg.lower():
        plot_html = get_mood_plot()
        emit('ai_response', plot_html)
        return
    if "schedule my work" in msg.lower():
        session['state'] = 'waiting_task_name'
        emit('ai_response', "Anchor: Let's schedule your task.")
        emit('ai_response', "Enter your task: ")
        return
    if msg.lower().endswith(" search"):
        query = msg[:-7].strip()
        results = search_web(query)
        result_text = format_search_results(results)
        emit('ai_response', result_text)
        if results:
            result_text_plain = "\n".join([f"{i+1}. {r['title']} - {r['url']}\n{r['description']}" for i, r in enumerate(results)])
            session['messages'].append({"role": "user", "content": f"I searched for '{query}'. Results:\n{result_text_plain}"})
            try:
                completion = client.chat.completions.create(
                    model="google/gemma-2-9b-it",
                    messages=session['messages']
                )
                response_text = completion.choices[0].message.get("content", "")
                emit('ai_response', f"Anchor: {response_text.strip()}")
                session['messages'].append({"role": "assistant", "content": response_text})
            except Exception:
                emit('ai_response', "Anchor: I found those results! Anything specific you'd like to know?")
        return
    if "daily affirmation" in msg.lower():
        emit('ai_response', f"Anchor: Here's your affirmation: {get_daily_affirmation()}")
        return
    if "study tips" in msg.lower() or "help me study" in msg.lower():
        tips = get_study_tips(mood)
        tip_text = "<p>Anchor: Study tips for your mood:</p><ul>" + "".join(f"<li>{i}. {tip}</li>" for i, tip in enumerate(tips, 1)) + "</ul>"
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
    if msg.lower() in ["suggest", "suggest songs", "suggest music", "suggest movies"]:
        suggest_type = "videos" if msg.lower() == "suggest" else msg.lower().split("suggest ")[1]
        recs = get_recommendations(mood, suggest_type=suggest_type)
        rec_text = format_recommendations(recs)
        emit('ai_response', rec_text)
        return

    if random.random() < 0.25:
        emotional_check_ins = [
            "I'm sensing something in your message - how are you really feeling?",
            "You know I care about you, right? What's on your heart?"
        ]
        check_in = random.choice(emotional_check_ins)
        emit('ai_response', f"Anchor: {check_in}")
        session['messages'].append({"role": "assistant", "content": check_in})

    session['messages'].append({"role": "user", "content": msg})
    if score < -0.3:
        session['messages'].append({
            "role": "system",
            "content": f"User seems to be struggling (sentiment score: {score:.3f}). Respond with extra compassion."
        })
    elif score > 0.3:
        session['messages'].append({
            "role": "system",
            "content": f"User is in a positive mood (sentiment score: {score:.3f}). Encourage their progress."
        })

    if len(session['messages']) > 25:
        system_msgs = [msg for msg in session['messages'] if msg['role'] == 'system']
        recent_msgs = session['messages'][-20:]
        session['messages'] = system_msgs + recent_msgs

    try:
        completion = client.chat.completions.create(
            model="google/gemma-2-9b-it",
            messages=session['messages']
        )
        response_text = completion.choices[0].message.get("content", "").strip()
        if any(phrase in response_text.lower() for phrase in ["i don't know", "i'm not sure"]):
            search_results = search_and_fetch_content(msg, num_results=2)
            if search_results:
                combined_content = "\n\n".join([r["content"][:800] for r in search_results])
                session['messages'].append({
                    "role": "system",
                    "content": f"Found info:\n{combined_content}\nUse this naturally in your response."
                })
                completion = client.chat.completions.create(
                    model="google/gemma-2-9b-it",
                    messages=session['messages']
                )
                response_text = completion.choices[0].message.get("content", "").strip()
        emit('ai_response', f"Anchor: {response_text}")
        session['messages'].append({"role": "assistant", "content": response_text})
    except Exception:
        fallback_response = "I'm here for you, even if I hit a snag. What's on your mind?"
        emit('ai_response', f"Anchor: {fallback_response}")

@socketio.on('feature')
def handle_feature(feat):
    sid = request.sid
    mood = session.get('last_mood', "😐 Neutral")
    if feat == 'mood_analysis':
        emit('ai_response', get_mood_plot())
    elif feat == 'daily_affirmation':
        emit('ai_response', f"Anchor: Here's your affirmation: {get_daily_affirmation()}")
    elif feat == 'study_tips':
        tips = get_study_tips(mood)
        tip_text = "<p>Anchor: Study tips for your mood:</p><ul>" + "".join(f"<li>{i}. {tip}</li>" for i, tip in enumerate(tips, 1)) + "</ul>"
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
        emit('ai_response', "Anchor: Let's schedule your task.")
        emit('ai_response', "Enter your task: ")
    elif feat == 'check_tasks':
        emit('update_tasks', format_tasks())
    elif feat == 'book_suggestions':
        books = get_book_recommendations(mood)
        book_text = format_book_recommendations(books)
        emit('ai_response', book_text)
    # No need for a closing brace; Python uses indentation to define scope
