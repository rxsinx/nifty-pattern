"""
Telegram Bot Integration for Indian Equity Pattern Analyzer
=============================================================

This module provides a Telegram bot interface for the stock analysis system.
Users can send stock symbols and receive comprehensive analysis reports.

Features:
- Stock symbol input via Telegram
- Comprehensive PDF report generation
- Pattern detection results
- Quantitative analysis
- Fractal price forecasts
- Chart images sent directly to Telegram

Author: Market Analyzer Pro
Version: 1.0 - Telegram Bot Integration
"""

import os
import io
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Telegram imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

# Analysis imports
import yfinance as yf
from pattern_detector import PatternDetector
from quantitative_analysis import FractalAnalysis, StatisticalEstimation, VolatilityModelling

# Plotting imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramStockBot:
    """
    Telegram bot for stock analysis and pattern detection.
    """
    
    def __init__(self, token: str):
        """
        Initialize Telegram bot.
        
        Args:
            token: Telegram Bot API token
        """
        self.token = token
        self.application = Application.builder().token(token).build()
        
        # Add handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup command and message handlers."""
        # Commands
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("analyze", self.analyze_command))
        self.application.add_handler(CommandHandler("patterns", self.patterns_list_command))
        self.application.add_handler(CommandHandler("forecast", self.forecast_command))
        
        # Message handler for stock symbols
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Callback query handler for inline buttons
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        welcome_message = """
🚀 **Welcome to Indian Equity Pattern Analyzer Bot!**

I can analyze Indian stocks and provide:
📊 30 Pattern Detection Methods
🔬 13 Quantitative Analysis Tools
🔮 30-Day Fractal Price Forecast
📈 Technical Indicators
📉 Risk Management Metrics

**How to use:**
1️⃣ Send me a stock symbol (e.g., RELIANCE, TCS, INFY)
2️⃣ Wait for comprehensive analysis
3️⃣ Receive detailed PDF report + charts

**Commands:**
/analyze SYMBOL - Full analysis report
/patterns - List all 30 patterns
/forecast SYMBOL - 30-day price forecast
/help - Show this message

**Example:**
Just type: `RELIANCE` or `/analyze RELIANCE.NS`

Let's start! Send me a stock symbol 📈
        """
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = """
📚 **Help - Available Commands**

**Basic Commands:**
/start - Start the bot
/help - Show this help message
/analyze SYMBOL - Get full analysis
/patterns - List all detection patterns
/forecast SYMBOL - Get 30-day forecast

**How to Input Stock Symbol:**
✅ RELIANCE (NSE symbol)
✅ RELIANCE.NS (Yahoo Finance format)
✅ TCS.NS
✅ INFY.NS

**What You'll Get:**
1. 📊 Pattern Detection (Zanger, Qullamaggie, etc.)
2. 🔬 Fractal Analysis (Hurst Exponent)
3. 📉 Volatility Models (GARCH)
4. 🔮 30-Day Price Forecast
5. 📈 Technical Indicators
6. 📄 Detailed PDF Report
7. 📊 Chart Images

**Example Usage:**
`/analyze RELIANCE.NS`
`/forecast TCS.NS`

Or simply type: `RELIANCE`
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def patterns_list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /patterns command - list all 30 patterns."""
        patterns_text = """
📚 **All 30 Pattern Detection Methods**

**🎯 Dan Zanger Patterns (6):**
1. Cup and Handle
2. High Tight Flag
3. Ascending Triangle
4. Flat Base
5. Falling Wedge
6. Double Bottom

**📊 Classic Patterns (8):**
7. Head & Shoulders 🔻
8. Double Top 🔻
9. Descending Triangle 🔻
10. Symmetrical Triangle
11. Bull Flag
12. Bear Flag 🔻
13. Rising Wedge 🔻
14. Pennant

**⚡ Qullamaggie Patterns (5):**
15. Episodic Pivot
16. Breakout
17. Parabolic Short 🔻
18. Gap and Go
19. ABCD Pattern

**🚀 Advanced Patterns (11):**
20. VCP 🔥 (Minervini)
21. Darvas Box 📦
22. Wyckoff Accumulation 📊
23. Wyckoff Distribution 🔻
24. CANSLIM Setup 💎
25. Inverse H&S 🔄
26. Triple Top 🔻🔻🔻
27. Triple Bottom 💚💚💚
28. Order Blocks 🏦 (ICT)
29. Elliott Wave 🌊
30. Mean Reversion 📉📈

🔻 = SHORT Signal
🔥 = Trend Following
📦 = Breakout
🔬 = Quantitative
        """
        await update.message.reply_text(patterns_text, parse_mode='Markdown')
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command."""
        if not context.args:
            await update.message.reply_text(
                "❌ Please provide a stock symbol.\n\nExample: `/analyze RELIANCE.NS`",
                parse_mode='Markdown'
            )
            return
        
        symbol = context.args[0].upper()
        await self.process_stock_analysis(update, symbol)
    
    async def forecast_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /forecast command - show only fractal forecast."""
        if not context.args:
            await update.message.reply_text(
                "❌ Please provide a stock symbol.\n\nExample: `/forecast RELIANCE.NS`",
                parse_mode='Markdown'
            )
            return
        
        symbol = context.args[0].upper()
        await self.process_forecast(update, symbol)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages (stock symbols)."""
        symbol = update.message.text.upper().strip()
        
        # Add .NS if not present
        if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
            symbol = symbol + '.NS'
        
        await self.process_stock_analysis(update, symbol)
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data.startswith('analyze_'):
            symbol = data.replace('analyze_', '')
            await self.process_stock_analysis(query, symbol)
        elif data.startswith('forecast_'):
            symbol = data.replace('forecast_', '')
            await self.process_forecast(query, symbol)
    
    async def process_stock_analysis(self, update: Update, symbol: str):
        """
        Process full stock analysis and send report.
        
        Args:
            update: Telegram update object
            symbol: Stock symbol to analyze
        """
        # Send processing message
        processing_msg = await update.message.reply_text(
            f"🔄 **Analyzing {symbol}...**\n\n"
            "⏳ Downloading data...\n"
            "⏳ Running pattern detection...\n"
            "⏳ Computing quantitative models...\n"
            "⏳ Generating forecast...\n"
            "⏳ Creating charts...\n"
            "⏳ Building PDF report...\n\n"
            "*This may take 30-60 seconds...*",
            parse_mode='Markdown'
        )
        
        try:
            # 1. Download data
            await processing_msg.edit_text(
                f"🔄 **Analyzing {symbol}...**\n\n"
                "✅ Downloading data...\n"
                "⏳ Running pattern detection...\n"
                "⏳ Computing quantitative models...\n"
                "⏳ Generating forecast...\n"
                "⏳ Creating charts...\n"
                "⏳ Building PDF report...",
                parse_mode='Markdown'
            )
            
            data = self.download_stock_data(symbol)
            
            if data is None or len(data) < 100:
                await processing_msg.edit_text(
                    f"❌ **Error**: Could not fetch data for {symbol}\n\n"
                    "Please check:\n"
                    "• Symbol is correct (e.g., RELIANCE.NS)\n"
                    "• Stock is listed on NSE/BSE\n"
                    "• Has sufficient trading history",
                    parse_mode='Markdown'
                )
                return
            
            # 2. Pattern Detection
            await processing_msg.edit_text(
                f"🔄 **Analyzing {symbol}...**\n\n"
                "✅ Downloading data...\n"
                "✅ Running pattern detection...\n"
                "⏳ Computing quantitative models...\n"
                "⏳ Generating forecast...\n"
                "⏳ Creating charts...\n"
                "⏳ Building PDF report...",
                parse_mode='Markdown'
            )
            
            patterns = self.detect_patterns(data)
            
            # 3. Quantitative Analysis
            await processing_msg.edit_text(
                f"🔄 **Analyzing {symbol}...**\n\n"
                "✅ Downloading data...\n"
                "✅ Running pattern detection...\n"
                "✅ Computing quantitative models...\n"
                "⏳ Generating forecast...\n"
                "⏳ Creating charts...\n"
                "⏳ Building PDF report...",
                parse_mode='Markdown'
            )
            
            quant_analysis = self.run_quantitative_analysis(data)
            
            # 4. Forecast
            await processing_msg.edit_text(
                f"🔄 **Analyzing {symbol}...**\n\n"
                "✅ Downloading data...\n"
                "✅ Running pattern detection...\n"
                "✅ Computing quantitative models...\n"
                "✅ Generating forecast...\n"
                "⏳ Creating charts...\n"
                "⏳ Building PDF report...",
                parse_mode='Markdown'
            )
            
            forecast = self.generate_forecast(data)
            
            # 5. Generate Charts
            await processing_msg.edit_text(
                f"🔄 **Analyzing {symbol}...**\n\n"
                "✅ Downloading data...\n"
                "✅ Running pattern detection...\n"
                "✅ Computing quantitative models...\n"
                "✅ Generating forecast...\n"
                "✅ Creating charts...\n"
                "⏳ Building PDF report...",
                parse_mode='Markdown'
            )
            
            chart_images = self.generate_charts(data, forecast, symbol)
            
            # 6. Generate PDF Report
            await processing_msg.edit_text(
                f"🔄 **Analyzing {symbol}...**\n\n"
                "✅ Downloading data...\n"
                "✅ Running pattern detection...\n"
                "✅ Computing quantitative models...\n"
                "✅ Generating forecast...\n"
                "✅ Creating charts...\n"
                "✅ Building PDF report...",
                parse_mode='Markdown'
            )
            
            pdf_path = self.generate_pdf_report(
                symbol, data, patterns, quant_analysis, forecast, chart_images
            )
            
            # 7. Send Summary
            await processing_msg.edit_text(
                f"✅ **Analysis Complete for {symbol}!**",
                parse_mode='Markdown'
            )
            
            summary = self.create_summary_message(symbol, data, patterns, forecast)
            await update.message.reply_text(summary, parse_mode='Markdown')
            
            # 8. Send Charts
            for chart_name, chart_path in chart_images.items():
                with open(chart_path, 'rb') as photo:
                    await update.message.reply_photo(
                        photo=photo,
                        caption=f"📊 {chart_name}"
                    )
            
            # 9. Send PDF Report
            with open(pdf_path, 'rb') as pdf_file:
                await update.message.reply_document(
                    document=pdf_file,
                    filename=f"{symbol}_Analysis_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    caption=f"📄 **Comprehensive Analysis Report for {symbol}**\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            
            # Cleanup temporary files
            self.cleanup_files([pdf_path] + list(chart_images.values()))
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            await processing_msg.edit_text(
                f"❌ **Error analyzing {symbol}**\n\n"
                f"Error: {str(e)}\n\n"
                "Please try again or contact support.",
                parse_mode='Markdown'
            )
    
    async def process_forecast(self, update: Update, symbol: str):
        """Process and send only forecast."""
        processing_msg = await update.message.reply_text(
            f"🔮 Generating 30-day forecast for {symbol}...",
            parse_mode='Markdown'
        )
        
        try:
            data = self.download_stock_data(symbol)
            
            if data is None:
                await processing_msg.edit_text(f"❌ Could not fetch data for {symbol}")
                return
            
            forecast = self.generate_forecast(data)
            forecast_chart = self.generate_forecast_chart(data, forecast, symbol)
            
            # Send forecast message
            forecast_msg = self.create_forecast_message(symbol, data, forecast)
            await processing_msg.edit_text(forecast_msg, parse_mode='Markdown')
            
            # Send chart
            with open(forecast_chart, 'rb') as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=f"🔮 30-Day Fractal Forecast for {symbol}"
                )
            
            # Cleanup
            self.cleanup_files([forecast_chart])
            
        except Exception as e:
            logger.error(f"Error forecasting {symbol}: {e}")
            await processing_msg.edit_text(f"❌ Error generating forecast: {str(e)}")
    
    def download_stock_data(self, symbol: str, period: str = '1y') -> Optional[pd.DataFrame]:
        """Download stock data from Yahoo Finance."""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                return None
            
            # Add technical indicators (simplified)
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_patterns(self, data: pd.DataFrame) -> Dict:
        """Run pattern detection."""
        try:
            detector = PatternDetector(data)
            
            patterns = {
                'zanger': detector.detect_all_zanger_patterns(),
                'classic': detector.detect_all_classic_patterns(),
                'swing': detector.detect_all_swing_patterns(),
                'advanced': detector.detect_all_wyckoff_canslim_patterns()
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {'zanger': [], 'classic': [], 'swing': [], 'advanced': []}
    
    def run_quantitative_analysis(self, data: pd.DataFrame) -> Dict:
        """Run quantitative analysis."""
        try:
            fractal = FractalAnalysis(data)
            stats = StatisticalEstimation(data)
            vol = VolatilityModelling(data)
            
            return {
                'hurst': fractal.calculate_hurst_exponent(),
                'fractal_dim': fractal.calculate_fractal_dimension(),
                'mle_normal': stats.mle_normal_distribution(),
                'garch': vol.garch_volatility()
            }
            
        except Exception as e:
            logger.error(f"Error in quant analysis: {e}")
            return {}
    
    def generate_forecast(self, data: pd.DataFrame) -> Dict:
        """Generate 30-day fractal forecast."""
        try:
            fractal = FractalAnalysis(data)
            forecast = fractal.forecast_fractal_price(forecast_days=30)
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {}
    
    def generate_charts(self, data: pd.DataFrame, forecast: Dict, symbol: str) -> Dict[str, str]:
        """Generate chart images."""
        charts = {}
        
        try:
            # 1. Price Chart with Forecast
            charts['forecast'] = self.generate_forecast_chart(data, forecast, symbol)
            
            # 2. Technical Indicators
            charts['indicators'] = self.generate_indicators_chart(data, symbol)
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            return {}
    
    def generate_forecast_chart(self, data: pd.DataFrame, forecast: Dict, symbol: str) -> str:
        """Generate forecast chart image."""
        fig = go.Figure()
        
        # Historical prices
        historical = data.tail(60)
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        if forecast:
            fig.add_trace(go.Scatter(
                x=forecast['dates'],
                y=forecast['mean_forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast['dates'] + forecast['dates'][::-1],
                y=forecast['ci_upper_95'] + forecast['ci_lower_95'][::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                name='95% CI'
            ))
        
        fig.update_layout(
            title=f'{symbol} - 30 Day Fractal Forecast',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            height=500
        )
        
        # Save to file
        filepath = f'/tmp/forecast_{symbol}_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
        fig.write_image(filepath, width=1200, height=600)
        
        return filepath
    
    def generate_indicators_chart(self, data: pd.DataFrame, symbol: str) -> str:
        """Generate technical indicators chart."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price & Moving Averages', 'RSI'),
            row_heights=[0.7, 0.3]
        )
        
        # Price and SMA
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='black')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='red')
        ), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=700)
        
        filepath = f'/tmp/indicators_{symbol}_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
        fig.write_image(filepath, width=1200, height=700)
        
        return filepath
    
    def generate_pdf_report(self, symbol: str, data: pd.DataFrame, patterns: Dict,
                           quant_analysis: Dict, forecast: Dict, chart_images: Dict) -> str:
        """Generate comprehensive PDF report."""
        filepath = f'/tmp/report_{symbol}_{datetime.now().strftime("%Y%m%d%H%M%S")}.pdf'
        
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        elements.append(Paragraph(f"Stock Analysis Report: {symbol}", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Summary Section
        elements.append(Paragraph("Executive Summary", styles['Heading2']))
        
        current_price = data['Close'].iloc[-1]
        summary_data = [
            ['Metric', 'Value'],
            ['Current Price', f"₹{current_price:.2f}"],
            ['30-Day Target', f"₹{forecast.get('target_price_30d', 0):.2f}"],
            ['Expected Return', f"{forecast.get('expected_return', 0):.2f}%"],
            ['Hurst Exponent', f"{quant_analysis.get('hurst', {}).get('hurst_exponent', 0):.3f}"],
            ['Patterns Detected', str(sum([len(p) for p in patterns.values()]))],
        ]
        
        t = Table(summary_data, colWidths=[3*inch, 3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 20))
        
        # Patterns Section
        elements.append(Paragraph("Detected Patterns", styles['Heading2']))
        
        for category, category_patterns in patterns.items():
            if category_patterns:
                elements.append(Paragraph(f"{category.capitalize()} Patterns ({len(category_patterns)})", styles['Heading3']))
                for pattern in category_patterns[:5]:  # Top 5 per category
                    pattern_text = f"• {pattern.get('pattern', 'Unknown')} - {pattern.get('signal', 'NEUTRAL')}"
                    elements.append(Paragraph(pattern_text, styles['Normal']))
        
        elements.append(PageBreak())
        
        # Charts
        for chart_name, chart_path in chart_images.items():
            elements.append(Paragraph(f"{chart_name.capitalize()} Chart", styles['Heading2']))
            img = Image(chart_path, width=6*inch, height=3*inch)
            elements.append(img)
            elements.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(elements)
        
        return filepath
    
    def create_summary_message(self, symbol: str, data: pd.DataFrame, 
                               patterns: Dict, forecast: Dict) -> str:
        """Create summary message for Telegram."""
        current_price = data['Close'].iloc[-1]
        total_patterns = sum([len(p) for p in patterns.values()])
        
        message = f"""
📊 **Analysis Summary for {symbol}**

**Current Metrics:**
💰 Price: ₹{current_price:.2f}
📈 30D Target: ₹{forecast.get('target_price_30d', 0):.2f}
📊 Expected Return: {forecast.get('expected_return', 0):.2f}%
🎯 Direction: {forecast.get('direction', 'NEUTRAL')}

**Patterns Detected: {total_patterns}**
🎯 Zanger: {len(patterns.get('zanger', []))}
📊 Classic: {len(patterns.get('classic', []))}
⚡ Swing: {len(patterns.get('swing', []))}
🚀 Advanced: {len(patterns.get('advanced', []))}

**Forecast Confidence:** {forecast.get('confidence_level', 'MEDIUM')}

📄 See detailed PDF report attached below!
        """
        
        return message
    
    def create_forecast_message(self, symbol: str, data: pd.DataFrame, forecast: Dict) -> str:
        """Create forecast message."""
        current_price = data['Close'].iloc[-1]
        
        if forecast.get('direction') == 'BULLISH':
            direction_emoji = "📈"
        elif forecast.get('direction') == 'BEARISH':
            direction_emoji = "📉"
        else:
            direction_emoji = "📊"
        
        message = f"""
🔮 **30-Day Fractal Forecast for {symbol}**

**Current Price:** ₹{current_price:.2f}

**Forecast:**
{direction_emoji} Target: ₹{forecast.get('target_price_30d', 0):.2f}
📊 Expected Return: {forecast.get('expected_return', 0):.2f}%
✅ Best Case: ₹{forecast.get('best_case_30d', 0):.2f}
❌ Worst Case: ₹{forecast.get('worst_case_30d', 0):.2f}

**Analysis:**
🎲 Hurst: {forecast.get('hurst_exponent', 0):.3f}
📐 Fractal Dim: {forecast.get('fractal_dimension', 0):.3f}
🎯 Bias: {forecast.get('forecast_bias', 'UNKNOWN')}
📊 Direction: {forecast.get('direction', 'NEUTRAL')}
⭐ Confidence: {forecast.get('confidence_level', 'MEDIUM')}

Chart attached below! 📊
        """
        
        return message
    
    def cleanup_files(self, filepaths: List[str]):
        """Cleanup temporary files."""
        for filepath in filepaths:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.error(f"Error removing {filepath}: {e}")
    
    def run(self):
        """Run the bot."""
        logger.info("Starting Telegram bot...")
        self.application.run_polling()


# Main execution
if __name__ == "__main__":
    # Get token from environment variable
    TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not TOKEN:
        print("❌ Error: TELEGRAM_BOT_TOKEN environment variable not set!")
        print("Please set it using: export TELEGRAM_BOT_TOKEN='your_token_here'")
        exit(1)
    
    # Create and run bot
    bot = TelegramStockBot(TOKEN)
    bot.run()
