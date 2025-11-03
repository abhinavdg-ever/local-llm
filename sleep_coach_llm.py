"""
Sleep Coach LLM System - Demo Mode

This application implements a Sleep Coach LLM system that processes wearable sleep data,
generates personal insights, and provides research-backed guidance using OpenAI models.
The system follows a layered architecture with data processing, analytics, and LLM integration.
"""

import os
import json
import datetime
import mysql.connector
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import uuid
import math
# Math utilities for analytics
class MathUtils:
    @staticmethod
    def mean(values: List[float]) -> float:
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def median(values: List[float]) -> float:
        vals = sorted([v for v in values if v is not None])
        n = len(vals)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return float(vals[mid])
        return (vals[mid - 1] + vals[mid]) / 2.0

    @staticmethod
    def mode(values: List[float]) -> float:
        vals = [v for v in values if v is not None]
        if not vals:
            return 0.0
        freq = {}
        for v in vals:
            freq[v] = freq.get(v, 0) + 1
        return max(freq.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        vals = sorted([v for v in values if v is not None])
        if not vals:
            return 0.0
        p = max(0.0, min(100.0, p))
        if len(vals) == 1:
            return float(vals[0])
        rank = (p / 100.0) * (len(vals) - 1)
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        if low == high:
            return float(vals[low])
        weight = rank - low
        return float(vals[low] * (1 - weight) + vals[high] * weight)

    @staticmethod
    def summation(values: List[float]) -> float:
        return float(sum([v for v in values if v is not None]))

    @staticmethod
    def minimum(values: List[float]) -> float:
        vals = [v for v in values if v is not None]
        return float(min(vals)) if vals else 0.0

    @staticmethod
    def maximum(values: List[float]) -> float:
        vals = [v for v in values if v is not None]
        return float(max(vals)) if vals else 0.0

# Load environment variables from .env file if available
from dotenv import load_dotenv
load_dotenv()

# Configuration
class Config:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENV")
    PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
    MODEL_NAME = "gpt-4-turbo"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    
    # MySQL Database Configuration
    DB_CONFIG = {
        'host': '62.72.57.99', 
        'user': 'aabo', 
        'password': '3#hxFkBFKJ2Ph!$@', 
        'database': 'aaboRing10Jan'
    }
    # MySQL tables (per user schema)
    SLEEP_SUMMARY_TABLE = "ai_coach_modules_summary"
    SLEEP_DETAILS_TABLE = "ai_coach_daily_sleep_details"
    
    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY or not cls.PINECONE_API_KEY:
            raise ValueError("API keys not set. Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables.")
        print("Configuration validated successfully")

# 1. Raw Sleep Data Processing
@dataclass
class SleepData:
    user_id: str
    date: datetime.date
    sleep_duration: float  # in hours
    deep_sleep: float  # in hours
    rem_sleep: float  # in hours
    light_sleep: float  # in hours
    awake_time: float  # in hours
    heart_rate: List[int]  # beats per minute
    movement: List[float]  # movement intensity
    respiration: List[float]  # breaths per minute
    
    @classmethod
    def from_wearable_data(cls, raw_data: Dict) -> 'SleepData':
        """Process raw wearable data into structured SleepData"""
        # Implementation would depend on the specific wearable device data format
        return cls(
            user_id=raw_data.get('user_id'),
            date=datetime.datetime.fromisoformat(raw_data.get('date')).date(),
            sleep_duration=raw_data.get('sleep_duration', 0),
            deep_sleep=raw_data.get('deep_sleep', 0),
            rem_sleep=raw_data.get('rem_sleep', 0),
            light_sleep=raw_data.get('light_sleep', 0),
            awake_time=raw_data.get('awake_time', 0),
            heart_rate=raw_data.get('heart_rate', []),
            movement=raw_data.get('movement', []),
            respiration=raw_data.get('respiration', [])
        )

# 2. Pseudonymization & PII Redaction
class PrivacyProcessor:
    def __init__(self):
        self.user_id_mapping = {}  # In production, this would be a secure database
    
    def get_pseudo_id(self, user_id: str) -> str:
        """Convert user ID to pseudonymized ID"""
        if user_id not in self.user_id_mapping:
            self.user_id_mapping[user_id] = f"user_{uuid.uuid4().hex[:8]}"
        return self.user_id_mapping[user_id]
    
    def redact_pii(self, data: Dict) -> Dict:
        """Remove all personal identifiers from data"""
        redacted = data.copy()
        if 'user_id' in redacted:
            redacted['user_pseudo_id'] = self.get_pseudo_id(redacted['user_id'])
            del redacted['user_id']
        
        # Remove other PII fields that might be present
        pii_fields = ['name', 'email', 'phone', 'address', 'dob', 'ssn']
        for field in pii_fields:
            if field in redacted:
                del redacted[field]
                
        return redacted

# 3. Time-series Analytics Database
class AnalyticsDatabase:
    def __init__(self):
        self.db_config = Config.DB_CONFIG
        self.summary_table = Config.SLEEP_SUMMARY_TABLE
        self.details_table = Config.SLEEP_DETAILS_TABLE
        self.connection = None
        self.connect_to_db()
        self._summary_columns = None
        self._details_columns = None
        # Resolve actual table names if configured ones are absent
        if self.connection:
            self.summary_table = self._resolve_table(self.summary_table, fallback_contains=["summary"]) or self.summary_table
            self.details_table = self._resolve_table(self.details_table, fallback_contains=["detail", "details"]) or self.details_table
    
    def connect_to_db(self):
        """Connect to MySQL database"""
        try:
            # Add a short timeout so the app doesn't hang if DB is unreachable
            self.connection = mysql.connector.connect(
                **self.db_config,
                connection_timeout=5
            )
            print("Successfully connected to MySQL database")
        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL database: {err}")
            # Fallback to in-memory storage if database connection fails
            self.daily_summaries = {}

    def _resolve_table(self, preferred_name: str, fallback_contains: List[str]) -> Optional[str]:
        """Return preferred_name if exists; else find first table containing all tokens in fallback_contains."""
        try:
            cursor = self.connection.cursor()
            # Check preferred first
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=%s AND table_name=%s",
                (self.db_config.get('database'), preferred_name)
            )
            exists = cursor.fetchone()[0] > 0
            if exists:
                cursor.close()
                return preferred_name
            # Find alternative
            like_clause = " AND ".join(["LOWER(table_name) LIKE %s" for _ in fallback_contains])
            params = tuple([f"%{token.lower()}%" for token in fallback_contains])
            cursor.execute(
                f"SELECT table_name FROM information_schema.tables WHERE table_schema=%s AND {like_clause} ORDER BY table_name ASC",
                (self.db_config.get('database'), *params)
            )
            row = cursor.fetchone()
            cursor.close()
            if row:
                print(f"Resolved table for tokens {fallback_contains}: {row[0]}")
                return row[0]
        except Exception as e:
            print(f"Table resolution error: {e}")
        return None

    def _get_columns(self, table_name: str) -> List[str]:
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s",
                (self.db_config.get('database'), table_name)
            )
            cols = [r[0] for r in cursor.fetchall()]
            cursor.close()
            return cols
        except Exception as e:
            print(f"Column introspection error for {table_name}: {e}")
            return []

    def _pick_col(self, candidates: List[str], available: List[str]) -> Optional[str]:
        available_l = {c.lower(): c for c in available}
        for cand in candidates:
            if cand.lower() in available_l:
                return available_l[cand.lower()]
        return None
    
    def store_daily_summary(self, user_pseudo_id: str, date: datetime.date, metrics: Dict):
        """Store aggregated daily sleep metrics"""
        if not self.connection:
            # Fallback to in-memory storage
            key = f"{user_pseudo_id}_{date.isoformat()}"
            self.daily_summaries[key] = metrics
            return
        
        # In a real implementation, you would insert/update the database
        # This is a placeholder for demonstration
        print(f"Storing data for user {user_pseudo_id} on {date}")
    
    def get_user_data(self, user_id: str, start_date: datetime.date, end_date: datetime.date, allow_fallback: bool = True) -> List[Dict]:
        """Retrieve user data for a date range from MySQL using details table columns provided by user.
        If allow_fallback is True and no rows in range, fetch latest recent rows.
        """
        if not self.connection:
            # Fallback to in-memory storage if database connection failed
            results = []
            current_date = start_date
            while current_date <= end_date:
                key = f"{user_id}_{current_date.isoformat()}"
                if hasattr(self, 'daily_summaries') and key in self.daily_summaries:
                    results.append(self.daily_summaries[key])
                current_date += datetime.timedelta(days=1)
            return results
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            # Use exact columns from ai_coach_daily_sleep_details per provided schema
            details_query = f"""
            SELECT COALESCE(CAST(`date` AS DATE), DATE(`start_time_stamp_converted`)) AS date,
                   COALESCE(netDuration, 0) AS netSleepDuration,
                   COALESCE(numAwakenings, 0) AS numAwakenings,
                   COALESCE(awakeDuration, 0) AS awakeDurationMinutes,
                   COALESCE(remDuration, 0) AS remDuration,
                   COALESCE(deepDuration, 0) AS deepDuration,
                   COALESCE(lightDuration, 0) AS lightDuration
            FROM {self.details_table}
            WHERE customer_id = %s
              AND COALESCE(CAST(`date` AS DATE), DATE(`start_time_stamp_converted`)) >= %s
              AND COALESCE(CAST(`date` AS DATE), DATE(`start_time_stamp_converted`)) <= %s
            ORDER BY COALESCE(CAST(`date` AS DATE), DATE(`start_time_stamp_converted`)) ASC
            """
            cursor.execute(details_query, (user_id, start_date, end_date))
            details_rows = cursor.fetchall()
            if details_rows:
                cursor.close()
                print(f"Using details table {self.details_table} with fixed columns: date, netDuration, numAwakenings, awakeDuration")
                return details_rows

            if not allow_fallback:
                cursor.close()
                return []

            # Fallback: fetch latest 60 records regardless of date range
            latest_query = f"""
            SELECT COALESCE(CAST(`date` AS DATE), DATE(`start_time_stamp_converted`)) AS date,
                   COALESCE(netDuration, 0) AS netSleepDuration,
                   COALESCE(numAwakenings, 0) AS numAwakenings,
                   COALESCE(awakeDuration, 0) AS awakeDurationMinutes,
                   COALESCE(remDuration, 0) AS remDuration,
                   COALESCE(deepDuration, 0) AS deepDuration,
                   COALESCE(lightDuration, 0) AS lightDuration
            FROM {self.details_table}
            WHERE customer_id = %s
            ORDER BY COALESCE(CAST(`date` AS DATE), DATE(`start_time_stamp_converted`)) DESC
            LIMIT 60
            """
            cursor.execute(latest_query, (user_id,))
            latest_rows = cursor.fetchall()
            cursor.close()
            if latest_rows:
                print(f"Falling back to latest records for customer {user_id}: {len(latest_rows)} rows")
                return list(reversed(latest_rows))
            return []
        except mysql.connector.Error as err:
            print(f"Error retrieving data from MySQL: {err}")
            return []

    def customers_overview(self, limit: int = 20) -> List[Dict]:
        """Return recent customers with date ranges for diagnostics."""
        if not self.connection:
            return []
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = f"""
            SELECT customer_id,
                   MIN(`date`) AS first_date,
                   MAX(`date`) AS last_date,
                   COUNT(*) AS records
            FROM {self.details_table}
            GROUP BY customer_id
            ORDER BY last_date DESC
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except mysql.connector.Error as err:
            print(f"Error retrieving customers overview: {err}")
            return []
    
    def calculate_trends(self, user_id: str, days: int, allow_fallback: bool = True) -> Dict:
        """Calculate sleep trends over specified number of days"""
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days)
        print(f"Looking for data for user {user_id} from {start_date} to {today}")
        data = self.get_user_data(user_id, start_date, today, allow_fallback=allow_fallback)
        
        if not data:
            print(f"No data found for user {user_id} in the specified period, trying to get any available data")
            # If no data in the specified period, try to get any available data for this user
            data = self.get_user_data(user_id, datetime.date(2020, 1, 1), today, allow_fallback=allow_fallback)
            if not data:
                overview = self.customers_overview()
                suggestions = ", ".join([str(r.get('customer_id')) for r in overview[:8]]) if overview else "(no suggestions)"
                print(f"No data found for user {user_id} at all. Example customer_ids: {suggestions}")
                return {"error": f"No data for user {user_id}. Try one of: {suggestions}"}
        
        # Process the actual data from database using correct column names
        sleep_durations = []
        awakenings = []
        awake_durations = []
        rem_durations = []
        deep_durations = []
        light_durations = []
        dates = []
        
        for entry in data:
            # Map likely column names from summary/details into expected keys
            date_value = entry.get('date') or entry.get('Date') or entry.get('day')
            if isinstance(date_value, datetime.date):
                dates.append(date_value.isoformat())
            else:
                dates.append(str(date_value))

            # Collect raw values first; we'll normalize units after loop
            sleep_durations.append(entry.get('netSleepDuration', 0) or 0)
            awakenings.append(
                entry.get('numAwakenings') or entry.get('awakenings') or entry.get('awakenings_count') or 0
            )
            awake_durations.append(entry.get('awakeDurationMinutes', 0) or 0)
            rem_durations.append(entry.get('remDuration', 0) or 0)
            deep_durations.append(entry.get('deepDuration', 0) or 0)
            light_durations.append(entry.get('lightDuration', 0) or 0)

        # Heuristic unit normalization:
        # If median per-night sleep value > 24, assume minutes and convert to hours; otherwise assume hours
        def maybe_minutes_to_hours(series: List[float]) -> List[float]:
            med = MathUtils.median(series)
            if med and med > 24:
                return [(v or 0) / 60.0 for v in series]
            return series

        sleep_durations = maybe_minutes_to_hours(sleep_durations)
        rem_durations = maybe_minutes_to_hours(rem_durations)
        deep_durations = maybe_minutes_to_hours(deep_durations)
        light_durations = maybe_minutes_to_hours(light_durations)
        
        # Calculate averages and trends
        trends = {
            "average_sleep_duration": sum(sleep_durations) / len(sleep_durations) if sleep_durations else 0,
            "average_awakenings": sum(awakenings) / len(awakenings) if awakenings else 0,
            "average_awake_duration": sum(awake_durations) / len(awake_durations) if awake_durations else 0,
            "average_rem_duration": sum(rem_durations) / len(rem_durations) if rem_durations else 0,
            "average_deep_duration": sum(deep_durations) / len(deep_durations) if deep_durations else 0,
            "average_light_duration": sum(light_durations) / len(light_durations) if light_durations else 0,
            "sleep_duration_trend": sleep_durations,
            "awakenings_trend": awakenings,
            "awake_duration_trend": awake_durations,
            "rem_duration_trend": rem_durations,
            "deep_duration_trend": deep_durations,
            "light_duration_trend": light_durations,
            "dates": dates,
            "total_records": len(data)
        }
        
        return trends

# 4. Cohort Aggregates
class CohortAnalytics:
    def __init__(self, analytics_db: AnalyticsDatabase):
        self.analytics_db = analytics_db
        self.cohort_stats = {
            "sleep_duration": [6.5, 7.0, 7.2, 7.5, 7.8, 8.0, 8.2],
            "deep_sleep": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            "rem_sleep": [1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
        }
    
    def update_cohort_statistics(self):
        """Update anonymized cohort statistics"""
        print("Demo mode: Mocking cohort statistics update")
    
    def get_percentile(self, metric: str, value: float) -> float:
        """Get percentile of a value within the cohort"""
        if metric not in self.cohort_stats:
            return 50.0  # Default for demo
            
        # Simple percentile calculation for demo
        return 65.0  # Mock percentile

# 5. LangChain Structured Tool Agent
class SleepCoachAgent:
    def __init__(
        self, 
        analytics_db: AnalyticsDatabase,
        cohort_analytics: CohortAnalytics,
        vector_db: Any  # Will be VectorDatabase instance
    ):
        self.analytics_db = analytics_db
        self.cohort_analytics = cohort_analytics
        self.vector_db = vector_db
        
        try:
            # Use OpenAI API directly with requests
            import requests
            self.openai_api_key = Config.OPENAI_API_KEY
            self.model_name = Config.MODEL_NAME
            self.llm = type('obj', (object,), {
                'predict': self._call_openai_api
            })
            print(f"Successfully initialized OpenAI API client: {Config.MODEL_NAME}")
        except Exception as e:
            print(f"Error initializing OpenAI API: {e}")
            # Fallback to mock LLM
            print("Falling back to mock OpenAI ChatModel")
            self.llm = type('obj', (object,), {
                'predict': lambda prompt: f"[DEMO RESPONSE] Based on your sleep data, here's an analysis: Your sleep duration has been consistent around 7.5 hours, which is within the recommended range. Your deep sleep (1.2 hours) is slightly below optimal levels. Consider reducing screen time before bed to improve sleep quality."
            })
        
        # Define tools
        self.tools = [
            {
                "name": "AnalyticsTool",
                "func": self.get_personal_data,
                "description": "Fetches personal sleep metrics and trends from Analytics DB"
            },
            {
                "name": "CohortTool",
                "func": self.get_cohort_comparison,
                "description": "Compares user metrics with anonymized cohort data"
            },
            {
                "name": "KnowledgeTool",
                "func": self.get_knowledge,
                "description": "Searches vector DB for sleep-related knowledge and research"
            },
            {
                "name": "ChartTool",
                "func": self.format_chart_data,
                "description": "Formats data for chart generation"
            }
        ]
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call custom LLM API hosted at http://34.131.0.29:11434/api/generate"""
        try:
            import requests
            payload = {
                "model": "llama3",
                "prompt": prompt
            }
            response = requests.post(
                "http://34.131.0.29:11434/api/generate",
                json=payload,
                timeout=60,
                stream=True
            )
            if response.status_code == 200:
                # The API streams responses line by line, each line is a JSON object with a 'response' key
                # We'll concatenate all 'response' values
                result_text = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            obj = json.loads(line.decode("utf-8"))
                            if "response" in obj:
                                result_text += obj["response"]
                        except Exception:
                            continue
                return result_text.strip()
            else:
                print(f"LLM API error: {response.status_code} - {response.text}")
                return f"[API ERROR] Trouble connecting to LLM service. Status: {response.status_code}"
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return f"[ERROR] I encountered an error while processing your request: {str(e)}"
    
    def get_personal_data(self, query: str, customer_id: str, days: int = 7) -> Dict:
        """Analytics Tool: Get personal sleep data and trends
        Args:
            query: User's query (for context)
            customer_id: Original customer ID for database queries (NOT pseudonymized)
            days: Number of days to analyze
        """
        return self.analytics_db.calculate_trends(customer_id, days)
    
    def get_cohort_comparison(self, metric: str, value: float) -> Dict:
        """Cohort Tool: Compare with anonymized cohort data"""
        percentile = self.cohort_analytics.get_percentile(metric, value)
        return {
            "metric": metric,
            "value": value,
            "percentile": percentile
        }
    
    def get_knowledge(self, query: str) -> List[Dict]:
        """Knowledge Tool: Search vector DB for relevant information"""
        # This would use the vector DB to retrieve relevant documents
        return self.vector_db.similarity_search(query)
    
    def format_chart_data(self, data: Dict) -> Dict:
        """Chart Tool: Format data for chart generation"""
        # Convert data into a format suitable for chart generation
        return {
            "chart_type": "line",  # or other types based on data
            "data": data,
            "format": "json"
        }
    
    def process_query(self, customer_id: str, query: str, user_pseudo_id: str = None) -> Dict:
        """Process user query and determine query type
        Args:
            customer_id: Original customer ID for database queries (NOT pseudonymized)
            query: User's query string
            user_pseudo_id: Pseudonymized ID for logging/output (optional, defaults to customer_id)
        """
        if user_pseudo_id is None:
            user_pseudo_id = customer_id
        query_lower = query.lower()
        
        response = {
            "query": query,
            "response_type": None,
            "content": None,
            "charts": None
        }
        
        try:
            if any(keyword in query_lower for keyword in ["my sleep", "my data", "how did i sleep", "trends", "sleep trends", "average", "avg", "sum", "median", "mode", "percentile"]):
                # Personal data query
                days = 30  # Default to 30 days to catch more historical data
                if "last 30 days" in query_lower or "past 30 days" in query_lower:
                    days = 30
                elif "last 14 days" in query_lower or "past 14 days" in query_lower:
                    days = 14
                elif "last 7 days" in query_lower or "past 7 days" in query_lower:
                    days = 7
                else:
                    import re
                    m = re.search(r"last\s+(\d+)\s+days", query_lower)
                    if m:
                        try:
                            days = max(1, min(120, int(m.group(1))))
                        except Exception:
                            pass
                # Month keyword hint for historical queries
                if "june" in query_lower or "july" in query_lower or "march" in query_lower:
                    days = 90  # For historical queries, use 90 days
                    
                # For explicit last N days queries, do not allow fallback
                # Use customer_id (not pseudonymized) for database queries
                allow_fb = not ("last" in query_lower and "days" in query_lower)
                data = self.analytics_db.calculate_trends(customer_id, days, allow_fallback=allow_fb)
                # Annotate how many records and the range used
                if "dates" in data:
                    try:
                        response["debug"] = {
                            "records": data.get("total_records"),
                            "from": data.get("dates", [None])[0],
                            "to": data.get("dates", [None])[-1],
                            "fallback": allow_fb
                        }
                    except Exception:
                        pass
 
                if "error" in data:
                    response["response_type"] = "personal_data"
                    # Surface backend suggestion if provided
                    error_text = data.get("error") or f"I couldn't find any recent sleep data for you over the last {days} days. Please make sure your wearable device is syncing data properly."
                    response["content"] = error_text
                else:
                    chart_data = self.format_chart_data(data)

                    # Compute math metrics if requested
                    def bullets(lines):
                        return "\n".join([f"- {line}" for line in lines])

                    lines = []
                    today_str = datetime.date.today().isoformat()
                    trends = data
                    sleep = trends.get("sleep_duration_trend", [])
                    rem = trends.get("rem_duration_trend", [])
                    deep = trends.get("deep_duration_trend", [])
                    light = trends.get("light_duration_trend", [])

                    q = query_lower
                    if "average" in q or "avg" in q:
                        lines.append(f"Avg sleep: {MathUtils.mean(sleep):.2f} h")
                        if rem: lines.append(f"Avg REM: {MathUtils.mean(rem):.2f} h")
                        if deep: lines.append(f"Avg Deep: {MathUtils.mean(deep):.2f} h")
                        if light: lines.append(f"Avg Light: {MathUtils.mean(light):.2f} h")
                    if "sum" in q:
                        lines.append(f"Total sleep: {MathUtils.summation(sleep):.2f} h")
                    if "median" in q:
                        lines.append(f"Median sleep: {MathUtils.median(sleep):.2f} h")
                    if "mode" in q:
                        lines.append(f"Mode sleep: {MathUtils.mode(sleep):.2f} h")
                    if "percentile" in q:
                        import re
                        m = re.search(r"(\d{1,2}|100)\s*percentile", q)
                        p = float(m.group(1)) if m else 50.0
                        lines.append(f"Sleep {p:.0f}th pct: {MathUtils.percentile(sleep, p):.2f} h")

                    # Default bullets if nothing specific matched
                    if not lines:
                        lines = [
                            f"Window: last {days} days ending {today_str}",
                            f"Avg sleep: {trends.get('average_sleep_duration', 0):.2f} h",
                            f"Avg awakenings: {trends.get('average_awakenings', 0):.1f}",
                            f"Avg awake: {trends.get('average_awake_duration', 0):.1f} min",
                            f"Records: {trends.get('total_records', 0)}"
                        ]

                    response["response_type"] = "personal_data"
                    response["content"] = bullets(lines)
                    response["charts"] = chart_data
                    
            elif any(keyword in query_lower for keyword in ["compare", "percentile", "others", "average", "how do i compare"]):
                # Cohort comparison query
                # Use customer_id (not pseudonymized) for database queries
                recent_data = self.get_personal_data(query, customer_id, 7)

                # Progressive backoff on range: 7 -> 30 -> 90 -> any
                data_candidate = recent_data
                if "error" in data_candidate:
                    data_candidate = self.get_personal_data(query, customer_id, 30)
                if "error" in data_candidate:
                    data_candidate = self.get_personal_data(query, customer_id, 90)
                if "error" in data_candidate:
                    today = datetime.date.today()
                    data_candidate = self.analytics_db.get_user_data(customer_id, datetime.date(2020, 1, 1), today)
                    if isinstance(data_candidate, list) and data_candidate:
                        # Wrap into trends-like structure for downstream
                        # Reuse calculate_trends to ensure shape
                        data_candidate = self.analytics_db.calculate_trends(customer_id, (today - datetime.date(2020,1,1)).days)

                if "error" in data_candidate:
                    # Provide suggestions if available
                    overview = self.analytics_db.customers_overview()
                    suggestions = ", ".join([str(r.get('customer_id')) for r in overview[:8]]) if overview else "(no suggestions)"
                    response["response_type"] = "cohort_comparison"
                    response["content"] = f"I need your sleep data to compare. Try customer_id one of: {suggestions}"
                else:
                    avg_sleep = data_candidate.get('average_sleep_duration', 0)
                    avg_awakenings = data_candidate.get('average_awakenings', 0)

                    prompt = f"""
                    You are a sleep coach AI. Respond in 5-7 bullet points. Short lines only.
                    Start each line with "- ". Avoid paragraphs.

                    User average:
                    - Sleep duration: {avg_sleep:.2f} h
                    - Awakenings: {avg_awakenings:.1f}

                    Reference ranges:
                    - Recommended sleep: 7-9 h
                    - Typical awakenings: 1-3/night

                    User query: "{query}"

                    Provide:
                    - Comparison to ranges
                    - Approx percentile phrasing
                    - 1-2 strengths, 1-2 focus areas
                    - 2 actionable tips
                    - Encouraging closer
                    """

                    llm_response = self.llm.predict(prompt)
                    response["response_type"] = "cohort_comparison"
                    response["content"] = llm_response
                    
            else:
                # Knowledge-based query
                knowledge_results = self.get_knowledge(query)
                
                # Create context from knowledge results
                knowledge_context = "\n".join([f"- {result['content']}" for result in knowledge_results[:3]])
                
                prompt = f"""
                You are a sleep coach AI. Answer in 5-8 bullet points. Short lines.
                Start each line with "- ". Avoid long explanations.

                User query: "{query}"

                Relevant knowledge:
                {knowledge_context}

                Provide:
                - Direct answer first
                - 2-3 practical tips
                - 1-2 brief research references (name only)
                - Encouraging closer
                """
                
                llm_response = self.llm.predict(prompt)
                response["response_type"] = "knowledge_based"
                response["content"] = llm_response
                
        except Exception as e:
            print(f"Error processing query: {e}")
            response["response_type"] = "error"
            response["content"] = f"I encountered an error while processing your request: {str(e)}. Please try again."
            
        return response

# 6. Vector DB (Pinecone)
class VectorDatabase:
    def __init__(self):
        try:
            # Import required libraries
            from langchain.vectorstores import Pinecone
            from langchain.embeddings.openai import OpenAIEmbeddings
            import pinecone
            
            # Initialize Pinecone
            pinecone.init(
                api_key=Config.PINECONE_API_KEY,
                environment=Config.PINECONE_ENVIRONMENT
            )
            
            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=Config.OPENAI_API_KEY,
                model=Config.EMBEDDING_MODEL
            )
            
            # Connect to Pinecone index
            self.index_name = Config.PINECONE_INDEX_NAME
            
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                # Create index if it doesn't exist
                pinecone.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric="cosine"
                )
                print(f"Created new Pinecone index: {self.index_name}")
            
            # Initialize vectorstore
            self.index = pinecone.Index(self.index_name)
            self.vectorstore = Pinecone(self.index, self.embeddings.embed_query, "text")
            
            print(f"Successfully connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            print(f"Error initializing Vector Database: {e}")
            # Fallback to mock implementation
            self.embeddings = None
            self.index = None
            self.vectorstore = None
            print("Falling back to mock Vector Database")
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector database"""
        if not self.vectorstore:
            print(f"Mock: Adding {len(documents)} documents to vector database")
            return
            
        try:
            # Process documents for vectorstore
            texts = [doc['content'] for doc in documents]
            metadatas = [{'source': doc.get('source', 'unknown')} for doc in documents]
            
            # Add to vectorstore
            self.vectorstore.add_texts(texts, metadatas)
            print(f"Added {len(documents)} documents to Pinecone index")
        except Exception as e:
            print(f"Error adding documents to vector database: {e}")
    
    def similarity_search(self, query: str, k: int = 3):
        """Search for similar documents"""
        if not self.vectorstore:
            # Return mock results if vectorstore not available
            return [
                {
                    'content': "Adults need 7-9 hours of sleep per night for optimal health.",
                    'source': "National Sleep Foundation"
                },
                {
                    'content': "Deep sleep is crucial for physical recovery and immune function.",
                    'source': "Journal of Sleep Research"
                },
                {
                    'content': "REM sleep plays a vital role in memory consolidation and emotional processing.",
                    'source': "Neuroscience & Biobehavioral Reviews"
                }
            ]
            
        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search(query, k=k)
            
            # Format results
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'unknown')
                })
                
            return formatted_results
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            # Return mock results as fallback
            return [
                {
                    'content': "Adults need 7-9 hours of sleep per night for optimal health.",
                    'source': "National Sleep Foundation"
                },
                {
                    'content': "Deep sleep is crucial for physical recovery and immune function.",
                    'source': "Journal of Sleep Research"
                },
                {
                    'content': "REM sleep plays a vital role in memory consolidation and emotional processing.",
                    'source': "Neuroscience & Biobehavioral Reviews"
                }
            ]

# 7. Conversation & Logging
class ConversationLogger:
    def __init__(self):
        self.sessions = {}  # In production, this would be a database
    
    def log_interaction(self, user_pseudo_id: str, query: str, response: Dict):
        """Log user interaction with redacted data"""
        session_id = f"session_{user_pseudo_id}_{datetime.datetime.now().date().isoformat()}"
        
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        # Redact any potentially sensitive information from the response
        redacted_response = response.copy()
        # Implementation would depend on what needs to be redacted
        
        self.sessions[session_id].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "response": redacted_response
        })
    
    def get_session_history(self, user_pseudo_id: str) -> List[Dict]:
        """Get session history for a user"""
        session_id = f"session_{user_pseudo_id}_{datetime.datetime.now().date().isoformat()}"
        return self.sessions.get(session_id, [])

# Main Sleep Coach LLM Application
class SleepCoachLLM:
    def __init__(self):
        # Initialize components
        self.privacy_processor = PrivacyProcessor()
        self.analytics_db = AnalyticsDatabase()
        self.cohort_analytics = CohortAnalytics(self.analytics_db)
        self.vector_db = VectorDatabase()
        self.agent = SleepCoachAgent(
            self.analytics_db,
            self.cohort_analytics,
            self.vector_db
        )
        self.logger = ConversationLogger()
        
    def process_wearable_data(self, raw_data: Dict) -> None:
        """Process incoming wearable data"""
        # 1. Convert to structured format
        sleep_data = SleepData.from_wearable_data(raw_data)
        
        # 2. Pseudonymize user ID
        user_pseudo_id = self.privacy_processor.get_pseudo_id(sleep_data.user_id)
        
        # 3. Store in analytics database
        metrics = {
            "date": sleep_data.date.isoformat(),
            "sleep_duration": sleep_data.sleep_duration,
            "deep_sleep": sleep_data.deep_sleep,
            "rem_sleep": sleep_data.rem_sleep,
            "light_sleep": sleep_data.light_sleep,
            "awake_time": sleep_data.awake_time,
            "average_heart_rate": sum(sleep_data.heart_rate) / len(sleep_data.heart_rate) if sleep_data.heart_rate else 0,
            "average_respiration": sum(sleep_data.respiration) / len(sleep_data.respiration) if sleep_data.respiration else 0,
            "movement_index": sum(sleep_data.movement) / len(sleep_data.movement) if sleep_data.movement else 0
        }
        
        self.analytics_db.store_daily_summary(user_pseudo_id, sleep_data.date, metrics)
        
        # 4. Update cohort statistics
        self.cohort_analytics.update_cohort_statistics()
        
    def handle_user_query(self, user_id: str, query: str) -> Dict:
        """Handle user query about sleep data or knowledge
        Args:
            user_id: Original customer_id (passed directly to database, not pseudonymized)
            query: User's query string
        """
        # 1. Pseudonymize user ID (for logging/output only)
        user_pseudo_id = self.privacy_processor.get_pseudo_id(user_id)
        
        # 2. Process query through agent (pass original customer_id for DB queries)
        response = self.agent.process_query(user_id, query, user_pseudo_id)
        
        # 3. Log interaction (use pseudonymized ID for privacy)
        self.logger.log_interaction(user_pseudo_id, query, response)
        
        return response
    
    def add_knowledge_documents(self, documents: List[Dict]) -> None:
        """Add sleep-related documents to knowledge base"""
        self.vector_db.add_documents(documents)

# Example usage
def main():
    print("\n===== Sleep Coach LLM Demo =====\n")
    
    # Initialize Sleep Coach LLM
    sleep_coach = SleepCoachLLM()
    
    # Example: Add knowledge documents
    print("\n1. Adding knowledge documents to vector database...")
    sleep_coach.add_knowledge_documents([
        {
            "content": "Research shows that adults need 7-9 hours of sleep per night for optimal health.",
            "source": "National Sleep Foundation"
        },
        {
            "content": "Deep sleep is crucial for physical recovery and immune function.",
            "source": "Journal of Sleep Research"
        },
        {
            "content": "REM sleep plays a vital role in memory consolidation and emotional processing.",
            "source": "Neuroscience & Biobehavioral Reviews"
        }
    ])
    
    # Example: Process wearable data
    print("\n2. Processing sample wearable sleep data...")
    sleep_coach.process_wearable_data({
        "user_id": "user123",
        "date": "2023-06-01",
        "sleep_duration": 7.5,
        "deep_sleep": 1.2,
        "rem_sleep": 1.8,
        "light_sleep": 4.0,
        "awake_time": 0.5,
        "heart_rate": [60, 58, 62, 57, 59],
        "movement": [0.1, 0.2, 0.1, 0.3, 0.1],
        "respiration": [14, 15, 14, 13, 15]
    })
    
    # Example: Handle user queries
    print("\n3. Testing different types of user queries:")
    
    # Personal data query
    print("\n3.1 Personal data query:")
    personal_query = "How was my sleep over the last 7 days?"
    print(f"User query: '{personal_query}'")
    response = sleep_coach.handle_user_query("user123", personal_query)
    print(f"Response type: {response['response_type']}")
    print(f"Content: {response['content']}")
    
    # Cohort comparison query
    print("\n3.2 Cohort comparison query:")
    cohort_query = "How does my deep sleep compare to others?"
    print(f"User query: '{cohort_query}'")
    response = sleep_coach.handle_user_query("user123", cohort_query)
    print(f"Response type: {response['response_type']}")
    print(f"Content: {response['content']}")
    
    # Knowledge-based query
    print("\n3.3 Knowledge-based query:")
    knowledge_query = "What's the importance of REM sleep?"
    print(f"User query: '{knowledge_query}'")
    response = sleep_coach.handle_user_query("user123", knowledge_query)
    print(f"Response type: {response['response_type']}")
    print(f"Content: {response['content']}")
    
    print("\n===== Demo Complete =====")

if __name__ == "__main__":
    # Check if running as script or imported as module
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        # Running as web app, don't execute demo
        pass
    else:
        main()