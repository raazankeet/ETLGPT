"""
CSS styling for the NLP to SQL Gradio application.
Provides density-based responsive styling for different display sizes.
"""

def get_css_for_density(density: str = "M") -> str:
    """
    Generate CSS based on UI density preference.
    
    Args:
        density: One of "XS", "S", "M", "L", "XL" for different text sizes and spacing
        
    Returns:
        Complete CSS string for the application
    """
    sizes = {
        "XS": {"font": 10, "lh": 1.26, "pad": 4, "cell": 4},
        "S":  {"font": 12, "lh": 1.28, "pad": 5, "cell": 5},
        "M":  {"font": 13, "lh": 1.32, "pad": 6, "cell": 6},
        "L":  {"font": 14, "lh": 1.36, "pad": 7, "cell": 7},
        "XL": {"font": 16, "lh": 1.40, "pad": 8, "cell": 8},
    }
    s = sizes.get(density, sizes["M"])
    
    return f"""
/* Hide footer */
footer {{ display: none !important; visibility: hidden !important; }}

/* Density-driven typography and spacing */
:root {{ --font-size: {s['font']}px !important; }}
.gradio-container, .gradio-container * {{ font-size: {s['font']}px !important; }}
.markdown-body, .prose, .prose * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
/* Chatbot overrides across Gradio versions */
.gradio-container .chatbot, .gradio-container .chatbot * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
.gradio-container .message, .gradio-container .message * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
.gradio-container .wrap, .gradio-container .wrap * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
/* Extra fallbacks for different Gradio builds */
.gradio-container .chat, .gradio-container .chat * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
.gradio-container .chatbot-message, .gradio-container .chatbot-message * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
.gradio-container .message-wrap, .gradio-container .message-wrap * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
/* Gradio 6.x likely wrappers */
.gradio-container .gr-chatbot, .gradio-container .gr-chatbot * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
.gradio-container [data-testid="chatbot"] *, .gradio-container [data-testid^="message" ] * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
.gradio-container [class*="chatbot"], .gradio-container [class*="chatbot"] * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
.gradio-container [class*="message"], .gradio-container [class*="message"] * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
.gradio-container pre, .gradio-container code {{ font-size: {max(9, s['font']-1)}px !important; }}
label, .block h1, .block h2, .block h3 {{ margin: {max(2, s['pad']-2)}px 0 {max(1, s['pad']-3)}px 0; }}
button, .btn {{ padding: {s['pad']}px {s['pad']+3}px !important; }}
input, textarea, select {{ padding: {s['pad']}px {max(4, s['pad']+1)}px !important; font-size: {s['font']}px !important; }}
.tab-nav button {{ padding: {s['pad']}px {s['pad']+3}px !important; }}

/* Chatbot specific scaling */
#chatbot, #chatbot * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}
#chatbot .message, #chatbot .message * {{ font-size: {s['font']}px !important; line-height: {s['lh']} !important; }}

/* Header styling */
.app-header {{ position: sticky; top: 0; z-index: 9999; background: #0f172a; color: #fff; }}
.app-header a {{ color: #93c5fd; text-decoration: none; margin-left: 16px; }}
.app-header a:hover {{ text-decoration: underline; }}
.md-docs h2, .md-docs h3 {{ margin-top: {max(8, s['pad']+4)}px; }}

/* Results table compact + sticky header + scroll */
#results_table {{ max-height: 420px; height: 360px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 6px; }}
#results_table table {{ font-size: {max(10, s['font']-1)}px; border-collapse: separate; border-spacing: 0; width: 100%; }}
#results_table thead th {{ position: sticky; top: 0; background: #f8fafc; z-index: 2; box-shadow: 0 1px 0 #e5e7eb; }}
#results_table td, #results_table th {{ padding: {s['cell']}px {s['cell']+2}px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; }}
#results_table tr:nth-child(even) td {{ background: #fafafa; }}
#results_table tr:hover td {{ background: #f1f5f9; }}

/* Chat spinner overlay */
.chat-container {{ position: relative; }}
#chat_spinner {{ position: absolute; bottom: 10px; left: 10px; display: flex; align-items: center; gap: 8px; background: rgba(255,255,255,0.95); padding: 8px 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); z-index: 1000; pointer-events: none; }}
#chat_spinner .spinner-text {{ font-size: 14px; color: #3498db; font-weight: 500; }}
#chat_spinner .lds-ring {{ display: inline-block; position: relative; width: 20px; height: 20px; }}
#chat_spinner .lds-ring div {{ box-sizing: border-box; display: block; position: absolute; width: 16px; height: 16px; margin: 2px; border: 2px solid #3498db; border-radius: 50%; animation: lds-ring 0.6s cubic-bezier(0.5, 0, 0.5, 1) infinite; border-color: #3498db transparent transparent transparent; }}
#chat_spinner .lds-ring div:nth-child(1) {{ animation-delay: -0.225s; }}
#chat_spinner .lds-ring div:nth-child(2) {{ animation-delay: -0.15s; }}
#chat_spinner .lds-ring div:nth-child(3) {{ animation-delay: -0.075s; }}
@keyframes lds-ring {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}

/* Tighten card paddings */
.svelte-1ipelgc, .svelte-18u9q0f, .panel {{ padding: {max(6, s['pad']+2)}px !important; }}
"""
