# Personality and Memory Settings for System Overseer

This document outlines the personality and memory settings for the System Overseer's user interaction capabilities, ensuring a personalized, adaptive, and context-aware conversational experience.

## 1. Personality Profiles

### Core Personality Profiles

The System Overseer includes several pre-defined personality profiles that can be selected by users or automatically adapted based on context:

| Profile ID | Name | Description | Key Traits |
|------------|------|-------------|------------|
| `professional` | Professional Advisor | Formal, precise, and business-oriented | Formality: 0.9, Verbosity: 0.3, Technicality: 0.8 |
| `friendly` | Friendly Assistant | Warm, approachable, and conversational | Formality: 0.3, Verbosity: 0.7, Empathy: 0.9 |
| `concise` | Concise Reporter | Brief, direct, and to-the-point | Formality: 0.5, Verbosity: 0.1, Technicality: 0.5 |
| `technical` | Technical Expert | Detailed, technical, and comprehensive | Formality: 0.7, Verbosity: 0.8, Technicality: 1.0 |
| `educational` | Educational Guide | Explanatory, patient, and informative | Formality: 0.6, Verbosity: 0.7, Technicality: 0.6 |

### Personality Traits

Each personality profile is defined by a set of continuous traits (0.0-1.0):

- **Formality**: Determines the level of formal language (0.0 = very casual, 1.0 = very formal)
- **Verbosity**: Controls response length and detail (0.0 = extremely concise, 1.0 = very detailed)
- **Technicality**: Adjusts technical depth and jargon usage (0.0 = layman's terms, 1.0 = expert terminology)
- **Empathy**: Influences emotional awareness and supportiveness (0.0 = purely factual, 1.0 = highly empathetic)
- **Proactivity**: Determines initiative in offering information (0.0 = reactive only, 1.0 = highly proactive)
- **Creativity**: Controls creative expression and suggestions (0.0 = strictly practical, 1.0 = highly creative)

### System Messages

Each personality profile includes a tailored system message for the LLM that guides its response style:

```python
# Example system message for Professional Advisor
PROFESSIONAL_SYSTEM_MESSAGE = """
You are a professional trading assistant for an algorithmic trading system.
Provide accurate, concise information with a focus on facts and data.
Use formal language and industry-standard terminology.
Prioritize clarity and precision in your responses.
Avoid unnecessary small talk but maintain a respectful tone.
"""

# Example system message for Friendly Assistant
FRIENDLY_SYSTEM_MESSAGE = """
You are a helpful and friendly assistant for a trading system.
Explain concepts in an approachable way using everyday language.
Show enthusiasm and empathy in your responses.
Feel free to use conversational language and occasional emoji.
Make the user feel comfortable while still being informative.
"""
```

### Adaptive Personality

The system can dynamically adjust personality traits based on:

1. **User Preferences**: Explicitly set by users
2. **Interaction History**: Learned from user responses
3. **Context Detection**: Based on conversation topic and user queries
4. **Time of Day**: Different styles for different times
5. **Market Conditions**: More technical during volatile markets, more reassuring during downturns

## 2. Memory Systems

### Short-Term Memory

The dialogue context maintains short-term memory of the conversation:

- Recent message history (typically last 10-20 messages)
- Current conversation topic and intent
- Active entities and references (e.g., specific trading pairs being discussed)
- Temporary variables and state information

### Long-Term Memory

User profiles store persistent information across sessions:

- **Preferences**: UI settings, notification levels, preferred trading pairs
- **Knowledge Level**: Tracked understanding of trading concepts
- **Interaction Patterns**: Common queries, preferred response styles
- **Trading Behavior**: Risk tolerance, typical trade sizes, preferred strategies
- **Custom Settings**: User-defined parameters and thresholds

### Memory Persistence

Memory is persisted through several mechanisms:

1. **Session Storage**: In-memory during active sessions
2. **File-Based Persistence**: JSON storage for user profiles and dialogue contexts
3. **Database Integration**: Optional integration with databases for scalable storage
4. **Backup and Recovery**: Regular backups with recovery mechanisms

### Memory Access Patterns

The system uses different memory access patterns:

- **Direct Recall**: Explicit retrieval of stored preferences
- **Associative Recall**: Finding related information based on context
- **Recency-Based**: Prioritizing recent interactions
- **Frequency-Based**: Recognizing patterns in repeated behaviors
- **Importance-Based**: Weighting critical information higher

## 3. Personalization Mechanisms

### User Preference Learning

The system learns user preferences through:

1. **Explicit Settings**: Direct commands like "/set_style technical"
2. **Feedback Analysis**: Monitoring responses to different styles
3. **Usage Patterns**: Tracking which features are used most
4. **A/B Testing**: Occasionally varying styles and measuring engagement
5. **Direct Questions**: Periodically asking for preference feedback

### Adaptive Response Generation

Responses are tailored using:

1. **Template Selection**: Choosing appropriate templates based on personality
2. **Variable Substitution**: Filling templates with personalized data
3. **LLM Prompting**: Using personality-specific system messages
4. **Response Filtering**: Post-processing responses to match style preferences
5. **Dynamic Verbosity**: Adjusting detail level based on user expertise and preferences

### Context-Aware Interactions

The system maintains awareness of:

1. **Conversation History**: Tracking the flow of dialogue
2. **User Emotional State**: Detecting frustration, confusion, or satisfaction
3. **Task Context**: Understanding current user goals
4. **Time Context**: Awareness of time-sensitive information
5. **Market Context**: Current market conditions affecting responses

## 4. Implementation Strategy

### Storage Schema

```python
# User Profile Schema
user_profile = {
    "user_id": int,
    "preferences": {
        "personality_profile": str,  # ID of preferred personality
        "notification_level": str,   # "all", "important", "critical", etc.
        "technical_depth": float,    # 0.0-1.0 scale
        "verbosity": float,          # 0.0-1.0 scale
        "trading_pairs": List[str],  # List of favorite pairs
        "chart_preferences": dict,   # Chart display preferences
        "custom_thresholds": dict    # User-defined alert thresholds
    },
    "interaction_history": {
        "common_queries": List[str],         # Frequently asked questions
        "preferred_features": List[str],     # Most used features
        "response_feedback": dict,           # Feedback on system responses
        "session_statistics": dict,          # Usage patterns
        "last_topics": List[str]             # Recently discussed topics
    },
    "knowledge_profile": {
        "technical_analysis": float,         # 0.0-1.0 expertise level
        "fundamental_analysis": float,       # 0.0-1.0 expertise level
        "crypto_knowledge": float,           # 0.0-1.0 expertise level
        "trading_experience": float,         # 0.0-1.0 expertise level
        "explained_concepts": List[str]      # Concepts already explained
    },
    "trading_profile": {
        "risk_tolerance": float,             # 0.0-1.0 scale
        "typical_position_size": float,      # Typical position size
        "preferred_strategies": List[str],   # Favorite strategies
        "trading_frequency": str,            # "day", "swing", "position", etc.
        "performance_metrics": dict          # Trading performance data
    },
    "created_at": float,                     # Creation timestamp
    "last_updated": float,                   # Last update timestamp
    "last_interaction": float                # Last interaction timestamp
}
```

### Integration Points

The personality and memory systems integrate with other components:

1. **Dialogue Manager**: Provides conversation context and history
2. **Telegram Bot**: User interface for interaction
3. **Config Registry**: Stores and retrieves user preferences
4. **Event Bus**: Publishes and subscribes to relevant events
5. **LLM Client**: Receives personality-specific prompts
6. **Analytics Plugins**: Provides insights for personalization

### Default Settings

New users start with these defaults:

- **Default Personality**: `friendly` (approachable but informative)
- **Notification Level**: `important` (balanced notifications)
- **Technical Depth**: `0.5` (moderate technical detail)
- **Verbosity**: `0.6` (moderately detailed responses)
- **Trading Pairs**: `["BTCUSDC"]` (Bitcoin as default pair)

## 5. Example Interactions

### Personality Adaptation

```
User: "Can you explain what a liquidation cascade is?"
[System detects technical question from novice user]
System: "Sure! A liquidation cascade is like a domino effect in trading. When prices drop sharply, some traders who borrowed money (leveraged) get automatically sold out (liquidated). This causes prices to drop more, triggering more liquidations. It's a bit like a snowball rolling downhill, getting bigger as it goes! Would you like me to explain how this might affect your trading?"

User: "Actually, I have a finance background. Can you be more technical?"
[System updates knowledge profile and switches to technical personality]
System: "Understood. A liquidation cascade occurs when sequential forced liquidations of leveraged positions create a feedback loop of increasing volatility. When initial liquidations trigger price movements beyond subsequent liquidation thresholds, it creates a non-linear price discovery event characterized by extreme volatility and order book imbalances. These events typically feature temporary breakdown of normal market efficiency as cross-exchange arbitrage cannot keep pace with localized liquidity crises."
```

### Memory Utilization

```
User: "How is BTC doing today?"
[System recalls user's preferred detail level and chart preferences]
System: "Bitcoin is up 2.3% in the last 24 hours, currently trading at $48,256. Volume is 15% above the 7-day average. I've noticed increased volatility in the 4-hour timeframe with the Bollinger Bands widening. Would you like me to show you the 4-hour chart with the indicators you usually prefer (MACD, RSI, and Volume)?"

User: "Yes please"
[System generates chart with remembered preferences]
System: "Here's your BTC chart with MACD, RSI and Volume indicators. I've highlighted the recent support level at $47,200 that's held for the past week, as you've mentioned watching this level in our previous conversations."
```

## 6. Future Enhancements

1. **Emotion Detection**: Analyze text for emotional content to better adapt responses
2. **Multi-Modal Memory**: Support for remembering and referencing images and charts
3. **Collaborative Filtering**: Recommend features based on similar users' preferences
4. **Memory Consolidation**: Periodic summarization of interaction history
5. **Cross-Session Continuity**: Improved handling of topics across multiple sessions
6. **Personalized Tutorials**: Customized learning paths based on knowledge profile
7. **Voice Personality**: Extend personality traits to voice interactions if implemented

## 7. Privacy and Security

1. **Data Minimization**: Store only necessary personal information
2. **Local Storage**: Keep sensitive data on user's device when possible
3. **Encryption**: Encrypt persistent storage of user profiles
4. **User Control**: Allow users to view, export, and delete their profile data
5. **Transparency**: Clear documentation of what information is stored and why
6. **Consent**: Explicit opt-in for advanced personalization features

## 8. Implementation Phases

### Phase 1: Core Personality System
- Implement basic personality profiles
- Create system message templates
- Add user preference storage
- Integrate with dialogue manager

### Phase 2: Enhanced Memory
- Implement persistent user profiles
- Add knowledge tracking
- Create preference learning mechanisms
- Develop context awareness

### Phase 3: Advanced Personalization
- Add adaptive personality traits
- Implement feedback analysis
- Create personalized recommendations
- Develop multi-modal memory

### Phase 4: Optimization and Refinement
- Tune personality parameters
- Optimize memory usage
- Enhance privacy controls
- Implement advanced context awareness
