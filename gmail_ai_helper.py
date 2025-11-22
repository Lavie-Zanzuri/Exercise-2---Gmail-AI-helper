#!/usr/bin/env python3
"""
Gmail AI Helper - Exercise 2 - Fixed Version
Analyzes Gmail emails using local LLM and caches results in Redis
"""

import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

# Third-party imports
import redis
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64
from email.mime.text import MIMEText

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalLLM:
    """Local LLM wrapper for Qwen model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate response from the local LLM"""
        try:
            # Format the prompt for chat
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            response = response[len(text):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error: Could not generate response"

class RedisCache:
    """Redis cache manager for LLM responses"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Could not connect to Redis: {e}")
            self.redis_client = None
    
    def get_cache_key(self, email_data: Dict) -> str:
        """Generate cache key for email analysis"""
        cache_string = f"{email_data['sender']}|{email_data['subject']}|{email_data.get('snippet', '')[:100]}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_cached_analysis(self, email_data: Dict) -> Optional[Dict]:
        """Get cached analysis for an email"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self.get_cache_key(email_data)
            cached_data = self.redis_client.get(f"email_analysis:{cache_key}")
            
            if cached_data:
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached analysis: {e}")
            return None
    
    def cache_analysis(self, email_data: Dict, analysis: Dict) -> None:
        """Cache email analysis for 8 hours"""
        if not self.redis_client:
            return
        
        try:
            cache_key = self.get_cache_key(email_data)
            cache_data = {
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 8 hours (28800 seconds)
            self.redis_client.setex(
                f"email_analysis:{cache_key}",
                28800,
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.error(f"Error caching analysis: {e}")

class GmailAnalyzer:
    """Gmail email analyzer using local LLM"""
    
    # Gmail API scopes
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    
    def __init__(self):
        self.service = None
        self.llm = LocalLLM()
        self.cache = RedisCache()
        self.analyses = []
        
        # Setup Gmail API
        self._setup_gmail_api()
    
    def _setup_gmail_api(self):
        """Setup Gmail API authentication"""
        creds = None
        
        # Load existing credentials
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists('credentials.json'):
                    logger.error("credentials.json file not found. Please download it from Google Cloud Console.")
                    raise FileNotFoundError("credentials.json not found")
                
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail API setup completed")
    
    def get_emails(self, max_results: int = 50) -> List[Dict]:
        """Fetch recent emails from Gmail"""
        try:
            # Get message IDs
            results = self.service.users().messages().list(
                userId='me',
                maxResults=max_results,
                q='in:inbox'
            ).execute()
            
            messages = results.get('messages', [])
            logger.info(f"Found {len(messages)} emails")
            
            emails = []
            for message in messages:
                try:
                    # Get full message
                    msg = self.service.users().messages().get(
                        userId='me',
                        id=message['id'],
                        format='full'
                    ).execute()
                    
                    # Extract headers
                    headers = msg['payload'].get('headers', [])
                    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                    sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                    date_str = next((h['value'] for h in headers if h['name'] == 'Date'), '')
                    
                    # Get snippet (preview text)
                    snippet = msg.get('snippet', '')[:200]  # Limit to 200 chars for LLM
                    
                    email_data = {
                        'id': message['id'],
                        'subject': subject,
                        'sender': sender,
                        'date': date_str,
                        'snippet': snippet
                    }
                    
                    emails.append(email_data)
                    
                except Exception as e:
                    logger.error(f"Error processing message {message['id']}: {e}")
                    continue
            
            return emails
            
        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
            return []
    
    def analyze_email(self, email: Dict) -> Dict:
        """Analyze a single email using LLM with improved prompt"""
        # Check cache first
        cached_result = self.cache.get_cached_analysis(email)
        if cached_result:
            logger.info(f"Using cached analysis for email: {email['subject'][:50]}...")
            return cached_result['analysis']
        
        # Create improved prompt with system and user messages
        system_prompt = """You are a strict JSON data extraction engine.
Extract email category, priority, and response status. Output ONLY valid JSON.
Categories:
- Work: Job emails from colleagues/boss
- Shopping: Orders, receipts, shipping
- Personal: Friends, family
- Promotional: Ads, sales, marketing
- Newsletter: Digests, subscriptions
- Social: Social media alerts
Priority:
- Urgent: Deadline today
- Important: Real person asking question
- Normal: Regular email
- Low: Automated, no action needed
response_needed:
- "Yes" = Real person asking YOU a question
- "No" = Automated, marketing, or just FYI
Do not output markdown, conversational text, or explanations. Return only the JSON."""

        user_prompt = f"""### Example:
Email Details:
- From: amazon-auto-confirm@amazon.com
- Subject: Your order #402-999 has shipped
- Preview: Your package containing 'USB-C Cable' is on the way and will arrive tomorrow.
Response:
{{
    "reason": "Automated shipping confirmation for purchased item.",
    "category": "Shopping",
    "priority": "Normal",
    "response_needed": "No"
}}

### Current Task:
Email Details:
- From: {email['sender'][:100]}
- Subject: {email['subject'][:100]}
- Preview: {email['snippet'][:150]}
Response:"""
        
        try:
            # Use the improved prompt with system and user messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Format for chat template
            text = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Generate response with shorter length for better JSON
            response = self.llm.generate_response(text, max_length=80)
            
            logger.info(f"Raw LLM response: {response[:100]}...")
            
            # Enhanced JSON extraction
            try:
                # Multiple strategies to find JSON
                json_candidates = []
                
                # Strategy 1: Direct search for JSON structure
                if '{' in response and '}' in response:
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    json_candidates.append(response[start:end])
                
                # Strategy 2: Look for JSON in lines
                for line in response.split('\n'):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        json_candidates.append(line)
                
                # Strategy 3: Clean markdown artifacts
                clean_response = response
                if '```json' in clean_response:
                    clean_response = clean_response.split('```json')[1].split('```')[0].strip()
                elif '```' in clean_response:
                    clean_response = clean_response.split('```')[1].split('```')[0].strip()
                
                if '{' in clean_response and '}' in clean_response:
                    start = clean_response.find('{')
                    end = clean_response.rfind('}') + 1
                    json_candidates.append(clean_response[start:end])
                
                # Try parsing each candidate
                for candidate in json_candidates:
                    try:
                        analysis = json.loads(candidate)
                        
                        # Validate and normalize
                        if 'category' in analysis and 'priority' in analysis and 'response_needed' in analysis:
                            # Normalize response_needed
                            if isinstance(analysis['response_needed'], str):
                                analysis['response_needed'] = analysis['response_needed'].lower() == 'yes'
                            
                            # Cache the result
                            self.cache.cache_analysis(email, analysis)
                            logger.info(f"✅ Successfully parsed: {analysis}")
                            return analysis
                    except json.JSONDecodeError:
                        continue
                
                # If no valid JSON found, use smart fallback
                logger.warning("No valid JSON found, using smart fallback")
                return self._smart_fallback_analysis(email, response)
                    
            except Exception as e:
                logger.warning(f"JSON parsing failed: {e}")
                return self._smart_fallback_analysis(email, response)
                
        except Exception as e:
            logger.error(f"Error analyzing email: {e}")
            return self._basic_fallback_analysis(email)
    
    def _smart_fallback_analysis(self, email: Dict, llm_response: str) -> Dict:
        """Smart fallback that extracts info from LLM response and email content"""
        subject = email['subject'].lower()
        sender = email['sender'].lower()
        snippet = email['snippet'].lower()
        response = llm_response.lower()
        
        # Smart category detection
        category = "Other"
        if any(word in subject + sender for word in ['job', 'work', 'career', 'interview', 'position', 'company', 'colleague']):
            category = "Work"
        elif any(word in subject + sender for word in ['amazon', 'order', 'shipped', 'delivery', 'purchase', 'buy', 'shop']):
            category = "Shopping"
        elif any(word in subject + sender for word in ['sale', 'promo', 'discount', 'offer', 'deal', 'black friday']):
            category = "Promotional"
        elif any(word in subject + sender for word in ['newsletter', 'digest', 'weekly', 'monthly', 'news']):
            category = "Newsletter"
        elif any(word in sender for word in ['gmail', 'hotmail', 'yahoo', 'friend']):
            category = "Personal"
        elif any(word in subject + sender for word in ['bank', 'invoice', 'payment', 'bill', 'statement']):
            category = "Finance"
        
        # Smart priority detection
        priority = "Normal"
        if any(word in subject + snippet for word in ['urgent', 'asap', 'critical', 'emergency', 'deadline']):
            priority = "Urgent"
        elif any(word in subject + snippet for word in ['important', 'meeting', 'interview', 'deadline']):
            priority = "Important"
        elif any(word in subject + sender for word in ['newsletter', 'promo', 'auto', 'noreply']):
            priority = "Low"
        
        # Smart response detection
        response_needed = False
        if any(word in snippet for word in ['?', 'please', 'reply', 'respond', 'let me know', 'question', 'can you']):
            response_needed = True
        elif any(word in subject for word in ['invitation', 'meeting', 'interview', 'request', 'confirm']):
            response_needed = True
        elif any(word in sender for word in ['noreply', 'auto', 'notification']):
            response_needed = False
        
        analysis = {
            "category": category,
            "priority": priority,
            "response_needed": response_needed
        }
        
        # Cache the result
        self.cache.cache_analysis(email, analysis)
        logger.info(f"🤖 Smart fallback: {analysis}")
        return analysis
    
    def _basic_fallback_analysis(self, email: Dict) -> Dict:
        """Basic fallback analysis"""
        return {
            "category": "Other",
            "priority": "Normal",
            "response_needed": False
        }
    
    def analyze_all_emails(self, max_emails: int = 50) -> List[Dict]:
        """Analyze all emails and return results"""
        logger.info(f"Starting analysis of {max_emails} emails...")
        
        emails = self.get_emails(max_emails)
        if not emails:
            logger.error("No emails found")
            return []
        
        results = []
        for i, email in enumerate(emails, 1):
            logger.info(f"Analyzing email {i}/{len(emails)}: {email['subject'][:50]}...")
            
            analysis = self.analyze_email(email)
            
            result = {
                **email,
                **analysis
            }
            results.append(result)
            
            # Small delay to avoid overwhelming the LLM
            time.sleep(0.1)
        
        self.analyses = results
        logger.info(f"Completed analysis of {len(results)} emails")
        return results
    
    def generate_charts(self):
        """Generate visualization charts for email analysis"""
        if not self.analyses:
            logger.error("No analysis data available for charts")
            return
        
        # Prepare data
        categories = [a['category'] for a in self.analyses]
        priorities = [a['priority'] for a in self.analyses]
        response_needed = [a['response_needed'] for a in self.analyses]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gmail Email Analysis Dashboard', fontsize=16)
        
        # 1. Category distribution
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        ax1.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax1.set_title('Email Categories Distribution')
        
        # 2. Priority distribution
        priority_counts = {}
        for priority in priorities:
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        colors = ['red', 'orange', 'yellow', 'green']
        ax2.pie(priority_counts.values(), labels=priority_counts.keys(), 
                autopct='%1.1f%%', colors=colors[:len(priority_counts)])
        ax2.set_title('Email Priority Distribution')
        
        # 3. Response needed
        response_counts = {'Needs Response': sum(response_needed), 
                          'No Response': len(response_needed) - sum(response_needed)}
        
        ax3.pie(response_counts.values(), labels=response_counts.keys(), autopct='%1.1f%%')
        ax3.set_title('Response Requirements')
        
        # 4. Top senders analysis
        senders = [a['sender'].split('<')[0].strip() if '<' in a['sender'] else a['sender'] 
                  for a in self.analyses]
        sender_counts = {}
        for sender in senders:
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
        
        # Top 10 senders
        top_senders = dict(sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        ax4.barh(list(top_senders.keys()), list(top_senders.values()))
        ax4.set_title('Top 10 Email Senders')
        ax4.set_xlabel('Number of Emails')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('outputs/gmail_analysis_charts.png', dpi=300, bbox_inches='tight')
        logger.info("Charts saved to gmail_analysis_charts.png")
        plt.show()
    
    def print_summary(self):
        """Print analysis summary"""
        if not self.analyses:
            logger.error("No analysis data available")
            return
        
        print("\n" + "="*60)
        print("GMAIL EMAIL ANALYSIS SUMMARY")
        print("="*60)
        
        # Category breakdown
        categories = {}
        priorities = {}
        response_needed_count = 0
        
        for analysis in self.analyses:
            cat = analysis['category']
            pri = analysis['priority']
            resp = analysis['response_needed']
            
            categories[cat] = categories.get(cat, 0) + 1
            priorities[pri] = priorities.get(pri, 0) + 1
            if resp:
                response_needed_count += 1
        
        print(f"\nTotal emails analyzed: {len(self.analyses)}")
        
        print("\nCATEGORY BREAKDOWN:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.analyses)) * 100
            print(f"  {cat}: {count} emails ({percentage:.1f}%)")
        
        print("\nPRIORITY BREAKDOWN:")
        for pri, count in sorted(priorities.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.analyses)) * 100
            print(f"  {pri}: {count} emails ({percentage:.1f}%)")
        
        print(f"\nEMAILS REQUIRING RESPONSE: {response_needed_count} ({(response_needed_count/len(self.analyses)*100):.1f}%)")
        
        # Show some examples
        print("\nURGENT EMAILS:")
        urgent_emails = [a for a in self.analyses if a['priority'] == 'Urgent']
        for email in urgent_emails[:5]:  # Show first 5
            print(f"  - {email['subject'][:60]}... (from: {email['sender'].split('<')[0].strip()})")
        
        print("\nEMAILS NEEDING RESPONSE:")
        response_emails = [a for a in self.analyses if a['response_needed']]
        for email in response_emails[:5]:  # Show first 5
            print(f"  - {email['subject'][:60]}... (from: {email['sender'].split('<')[0].strip()})")
        
        print("\n" + "="*60)

def main():
    """Main function"""
    try:
        # Initialize the analyzer
        analyzer = GmailAnalyzer()
        
        # Analyze emails
        results = analyzer.analyze_all_emails(max_emails=50)
        
        if results:
            # Print summary
            analyzer.print_summary()
            
            # Generate charts
            analyzer.generate_charts()
            
            # Save results to JSON file
            with open('/mnt/user-data/outputs/email_analysis_results.json', 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info("Analysis completed successfully!")
            logger.info("Results saved to email_analysis_results.json")
            logger.info("Charts saved to gmail_analysis_charts.png")
        else:
            logger.error("No emails were analyzed")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()