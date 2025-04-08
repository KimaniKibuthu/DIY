import React, { useState, useRef, useEffect } from 'react';
import { Phone, Send, Menu, X, BarChart2, Calendar, TrendingUp, Users } from 'lucide-react';
import { 
  ResponsiveContainer, CartesianGrid, Tooltip, Legend, 
  BarChart, Bar, XAxis, YAxis
} from 'recharts';
import axios from 'axios';

// API base URL
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/';

// Configure axios with default settings
axios.defaults.timeout = 10000; // 10 seconds timeout
axios.interceptors.request.use(request => {
  console.log('Starting Request:', request.url);
  return request;
});

axios.interceptors.response.use(
  response => {
    console.log('Response:', response.status, response.data);
    return response;
  },
  error => {
    console.error('Axios Error:', error);
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('Response data:', error.response.data);
      console.error('Response status:', error.response.status);
      console.error('Response headers:', error.response.headers);
    } else if (error.request) {
      // The request was made but no response was received
      console.error('No response received:', error.request);
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('Request error:', error.message);
    }
    return Promise.reject(error);
  }
);

// Helper functions
const formatNumber = (num) => {
  return num ? Number(num).toLocaleString('en-US', { maximumFractionDigits: 2 }) : "0";
};

const formatDate = (dateString) => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
};

const extractDatesFromText = (text) => {
  const dateRegex = /\d{4}-\d{2}-\d{2}/g;
  return text.match(dateRegex) || [];
};

const parseDateRange = (text) => {
  const dates = extractDatesFromText(text);
  if (dates.length === 0) return null;
  
  if (dates.length === 1) {
    return {
      start_date: dates[0],
      end_date: dates[0]
    };
  }
  
  return {
    start_date: dates[0],
    end_date: dates[1]
  };
};

// Components
const LoadingIndicator = () => (
  <div className="flex space-x-1">
    <div className="h-2 w-2 bg-green-600 rounded-full animate-bounce"></div>
    <div className="h-2 w-2 bg-green-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
    <div className="h-2 w-2 bg-green-600 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
  </div>
);

const MessageBubble = ({ message }) => {
  const isUser = message.role === 'user';
  
  // Handle different response formats
  const renderContent = () => {
    if (typeof message.content === 'string') {
      return <p className={`text-base ${isUser ? 'text-white' : 'text-gray-800'}`}>{message.content}</p>;
    }
    
    // Handle causal analysis content
    if (message.content?.summary && message.content?.plots) {
        const { summary, seasonal_recommendations, non_seasonal_recommendations, plots } = message.content;
      
      return (
        <div className="space-y-4">
          <div className="font-semibold text-lg border-b border-gray-200 pb-2">
            Revenue Analysis {formatDate(summary.start_date)} - {formatDate(summary.end_date)}
          </div>
          
          <div className="grid grid-cols-2 gap-4 bg-gray-50 p-3 rounded-lg">
            <div>
              <p className="font-medium">Current Revenue</p>
              <p className="text-xl">₹ {formatNumber(summary.current_revenue)}</p>
            </div>
            <div>
              <p className="font-medium">Target Revenue</p>
              <p className="text-xl">₹ {formatNumber(summary.target_revenue)}</p>
            </div>
            <div>
              <p className="font-medium">Revenue Increase Needed</p>
              <p className="text-xl text-red-600">₹ {formatNumber(summary.revenue_increase_needed)}</p>
            </div>
            <div>
              <p className="font-medium">Estimated Increase</p>
              <p className="text-xl text-green-600">₹ {formatNumber(summary.total_estimated_increase)}</p>
            </div>
          </div>
          
          <div>
            <img src={`data:image/png;base64,${plots.revenue_plot}`} alt="Revenue Analysis" className="w-full rounded-lg shadow-md" />
          </div>
          
          <div className="bg-green-50 p-3 rounded-lg mt-4">
            <p className="font-semibold text-lg mb-2">Recommended Actions</p>
            
            {non_seasonal_recommendations.length > 0 && (
              <div className="mb-4">
                <p className="font-medium text-green-700 mb-2">Non-Seasonal Recommendations:</p>
                <img src={`data:image/png;base64,${plots.feature_effects_plot}`} alt="Feature Effects" className="w-full rounded-lg shadow-md mb-3" />
                <img src={`data:image/png;base64,${plots.recommendations_plot}`} alt="Recommendations" className="w-full rounded-lg shadow-md mb-3" />
                
                <div className="space-y-3 mt-2">
                  {non_seasonal_recommendations.map((rec, idx) => (
                    <div key={idx} className="bg-white p-3 rounded-lg shadow-sm">
                      <p className="font-medium">{rec.feature}</p>
                      <div className="grid grid-cols-2 gap-2 text-sm mt-1">
                        <div>Current: {formatNumber(rec.current_value)}</div>
                        <div>Target: {formatNumber(rec.recommended_value)}</div>
                        <div className="text-green-600">Increase: {formatNumber(rec.increase_by)}</div>
                        <div>Change: {rec.percent_increase.toFixed(2)}%</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {seasonal_recommendations.length > 0 && (
              <div>
                <p className="font-medium text-green-700 mb-2">Seasonal Strategies:</p>
                <div className="space-y-3">
                  {seasonal_recommendations.map((rec, idx) => (
                    <div key={idx} className="bg-white p-3 rounded-lg shadow-sm">
                      <p className="font-medium">{rec.feature}</p>
                      <ul className="list-disc pl-5 mt-1 text-sm">
                        {rec.recommendations.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      );
    }
    
    // Handle effect analysis content
    if (message.content?.effect !== undefined) {
      const { effect, standard_error, p_value, plot, date, action } = message.content;
      
      return (
        <div className="space-y-4">
          <div className="font-semibold text-lg border-b border-gray-200 pb-2">
            Effect Analysis: {action} on {formatDate(date)}
          </div>
          
          <div className="grid grid-cols-3 gap-4 bg-gray-50 p-3 rounded-lg">
            <div>
              <p className="font-medium">Effect</p>
              <p className="text-xl">₹ {formatNumber(effect)}</p>
            </div>
            <div>
              <p className="font-medium">Standard Error</p>
              <p className="text-xl">₹ {formatNumber(standard_error)}</p>
            </div>
            <div>
              <p className="font-medium">P-value</p>
              <p className="text-xl">{p_value.toFixed(4)}</p>
              <p className="text-xs">{p_value < 0.05 ? 'Significant' : 'Not significant'}</p>
            </div>
          </div>
          
          <div>
            <img src={`data:image/png;base64,${plot}`} alt="Effect Analysis" className="w-full rounded-lg shadow-md" />
          </div>
          
          <div className="bg-blue-50 p-3 rounded-lg">
            <p className="font-semibold">Interpretation</p>
            <p className="mt-1">
              {effect > 0 
                ? `The ${action} had a positive effect of ₹${formatNumber(effect)} on voice revenue.`
                : `The ${action} had a negative effect of ₹${formatNumber(Math.abs(effect))} on voice revenue.`
              }
              {p_value < 0.05
                ? ' This effect is statistically significant.'
                : ' However, this effect is not statistically significant.'
              }
            </p>
          </div>
        </div>
      );
    }
    
    // Handle multiple effects analysis content
    if (message.content?.results && Array.isArray(message.content.results)) {
      const { results } = message.content;
      
      return (
        <div className="space-y-4">
          <div className="font-semibold text-lg border-b border-gray-200 pb-2">
            Multiple Effects Analysis
          </div>
          
          {results.map((result, idx) => (
            <div key={idx} className="bg-white p-4 rounded-lg shadow-sm mb-4">
              {result.error ? (
                <p className="text-red-500">{result.error}</p>
              ) : (
                <>
                  <p className="font-medium">{formatDate(result.date)}</p>
                  
                  <div className="grid grid-cols-3 gap-4 bg-gray-50 p-3 rounded-lg my-2">
                    <div>
                      <p className="font-medium">Effect</p>
                      <p className="text-lg">₹ {formatNumber(result.effect)}</p>
                    </div>
                    <div>
                      <p className="font-medium">Standard Error</p>
                      <p className="text-lg">₹ {formatNumber(result.standard_error)}</p>
                    </div>
                    <div>
                      <p className="font-medium">P-value</p>
                      <p className="text-lg">{result.p_value.toFixed(4)}</p>
                      <p className="text-xs">{result.p_value < 0.05 ? 'Significant' : 'Not significant'}</p>
                    </div>
                  </div>
                  
                  <div>
                    <img src={`data:image/png;base64,${result.plot}`} alt={`Effect Analysis ${result.date}`} className="w-full rounded-lg shadow-md" />
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      );
    }
    
    // Handle forecast content
    if (message.content?.type === 'single_day') {
      const { date, forecasts } = message.content;
      
      const forecastData = Object.entries(forecasts).map(([year, value]) => ({
        year,
        value: value || 0
      }));
      
      return (
        <div className="space-y-4">
          <div className="font-semibold text-lg border-b border-gray-200 pb-2">
            Forecast for {formatDate(date)}
          </div>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={forecastData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis tickFormatter={(value) => `₹${(value/1000000).toFixed(1)}M`} />
                <Tooltip formatter={(value) => [`₹${formatNumber(value)}`, 'Revenue']} />
                <Legend />
                <Bar dataKey="value" name="Revenue" fill="#059669" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="bg-green-50 p-3 rounded-lg">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead>
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Year</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Forecast Value</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {Object.entries(forecasts).map(([year, value]) => (
                    <tr key={year}>
                      <td className="px-4 py-2 whitespace-nowrap">{year}</td>
                      <td className="px-4 py-2 whitespace-nowrap text-right">{value ? `₹ ${formatNumber(value)}` : 'No data'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      );
    }
    
    if (message.content?.type === 'date_range') {
      const { start_date, end_date, summary, plot } = message.content;
      
      return (
        <div className="space-y-4">
          <div className="font-semibold text-lg border-b border-gray-200 pb-2">
            Forecast from {formatDate(start_date)} to {formatDate(end_date)}
          </div>
          
          <div className="bg-green-50 p-3 rounded-lg">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead>
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Year</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Min</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Max</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Mean</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Sum</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {summary.map((item) => (
                    <tr key={item.Year}>
                      <td className="px-4 py-2 whitespace-nowrap">{item.Year}</td>
                      <td className="px-4 py-2 whitespace-nowrap text-right">₹ {formatNumber(item.Min)}</td>
                      <td className="px-4 py-2 whitespace-nowrap text-right">₹ {formatNumber(item.Max)}</td>
                      <td className="px-4 py-2 whitespace-nowrap text-right">₹ {formatNumber(item.Mean)}</td>
                      <td className="px-4 py-2 whitespace-nowrap text-right">₹ {formatNumber(item.Sum)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          {plot && (
            <div className="bg-white rounded-lg shadow-md p-2 h-80">
              <iframe
                title="Forecast Chart"
                srcDoc={`
                  <html>
                    <head>
                      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                      <style>body { margin: 0; }</style>
                    </head>
                    <body>
                      <div id="plot" style="width: 100%; height: 100%;"></div>
                      <script>
                        const plotData = ${JSON.stringify(plot)};
                        Plotly.newPlot('plot', plotData.data, plotData.layout, {responsive: true});
                      </script>
                    </body>
                  </html>
                `}
                width="100%"
                height="100%"
                frameBorder="0"
              />
            </div>
          )}
        </div>
      );
    }
    
    // Default for unhandled content types
    return <p className="text-gray-800">Unsupported response format</p>;
  };
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`
        max-w-2xl 
        rounded-xl 
        p-5 
        shadow-md
        ${isUser ? 
          'bg-gradient-to-r from-green-600 to-green-700 text-white ml-12' : 
          'bg-white bg-opacity-95 border border-gray-200 mr-12 shadow-lg'}
      `}>
        {!isUser && (
          <div className="flex items-start">
            <div className="flex-shrink-0 mr-3">
              <div className="h-10 w-10 rounded-full bg-green-100 flex items-center justify-center shadow-inner">
                <Phone className="h-6 w-6 text-green-700" />
              </div>
            </div>
            <div className="flex-1">
              {renderContent()}
            </div>
          </div>
        )}
        
        {isUser && renderContent()}
      </div>
    </div>
  );
};

const VoiceCausalUI = () => {
  const [messages, setMessages] = useState([
    { id: 1, role: "assistant", content: "Hello! I'm the Voice Causal Explainer. How can I help you analyze your voice revenue data today?" }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dataRange, setDataRange] = useState(null);
  
  // Fetch initial data range
  useEffect(() => {
    const fetchDataRange = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/data-range`);
        setDataRange(response.data);
      } catch (error) {
        console.error('Error fetching data range:', error);
      }
    };
    
    fetchDataRange();
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      processUserMessage(inputValue);
      setInputValue('');
    }
  };
  
  const handleExampleClick = (exampleText) => {
    processUserMessage(exampleText);
  };
  
  const processUserMessage = async (text) => {
    // Add user message to chat
    const userMessage = { id: messages.length + 1, role: "user", content: text };
    setMessages(prev => [...prev, userMessage]);
    setIsProcessing(true);
    
    try {
      // Determine query type and call appropriate API endpoint
      const lowerText = text.toLowerCase();
      
      // Cause analysis
      if (lowerText.includes('cause') || lowerText.includes('why') || lowerText.includes('explain') || lowerText.includes('recommend')) {
        const dateRange = parseDateRange(text);
        if (dateRange) {
          const response = await axios.post(`${API_BASE_URL}/api/cause`, dateRange);
          addAssistantMessage(response.data);
          return;
        }
      }
      
      // Effect analysis (single date)
      if (lowerText.includes('effect') && !lowerText.includes('effects')) {
        const dates = extractDatesFromText(text);
        if (dates.length > 0) {
          const actionMatch = text.match(/effect of (.*?) on/i);
          const action = actionMatch ? actionMatch[1] : "the event";
          
          const response = await axios.post(`${API_BASE_URL}/api/effect`, {
            date: dates[0],
            action: action
          });
          addAssistantMessage(response.data);
          return;
        }
      }
      
      // Multiple effects analysis
      if (lowerText.includes('effects')) {
        const dates = extractDatesFromText(text);
        if (dates.length > 0) {
          const response = await axios.post(`${API_BASE_URL}/api/effects`, {
            dates: dates
          });
          addAssistantMessage(response.data);
          return;
        }
      }
      
      // Forecast analysis
      if (lowerText.includes('forecast') || lowerText.includes('predict')) {
        const dates = extractDatesFromText(text);
        if (dates.length === 1) {
          const response = await axios.post(`${API_BASE_URL}/api/forecast`, {
            start_date: dates[0]
          });
          addAssistantMessage(response.data);
          return;
        } else if (dates.length >= 2) {
          const response = await axios.post(`${API_BASE_URL}/api/forecast`, {
            start_date: dates[0],
            end_date: dates[1]
          });
          addAssistantMessage(response.data);
          return;
        }
      }
      
      // For unrecognized queries
      addAssistantMessage(
        "I understand you want to analyze voice revenue data. Could you please specify your request with a date or date range? For example:\n\n" +
        "- What was the cause of revenue change on 2024-01-15?\n" +
        "- What was the effect of the price change on 2024-02-01?\n" +
        "- Forecast revenue for 2024-03-01 to 2024-03-31"
      );
      
    } catch (error) {
      console.error('Error processing query:', error);
      let errorMessage = "Sorry, there was an error processing your request.";
      
      if (error.response && error.response.data && error.response.data.error) {
        errorMessage = `Error: ${error.response.data.error}`;
      }
      
      addAssistantMessage(errorMessage);
    }
  };
  
  const addAssistantMessage = (content) => {
    const assistantMessage = { 
      id: messages.length + 1, 
      role: "assistant", 
      content 
    };
    setMessages(prev => [...prev, assistantMessage]);
    setIsProcessing(false);
  };

  return (
    <div className="flex h-screen bg-gradient-to-r from-green-50 via-gray-50 to-white text-gray-800 overflow-hidden">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed md:relative md:translate-x-0 z-10 h-full w-64 bg-white shadow-lg transform transition-transform duration-200 ease-in-out border-r border-green-100`}>
        <div className="p-4 border-b border-gray-200 flex items-center justify-between bg-green-700 text-white">
          <div className="flex items-center space-x-2">
            <div className="h-8 w-8 rounded-full bg-white flex items-center justify-center">
              <Phone className="h-5 w-5 text-green-700" />
            </div>
            <span className="font-bold text-lg">Voice Causal Explainer</span>
          </div>
          <button 
            onClick={() => setSidebarOpen(false)} 
            className="md:hidden p-1 rounded-full hover:bg-green-600"
          >
            <X className="h-5 w-5 text-white" />
          </button>
        </div>
        
        <div className="p-4 overflow-y-auto h-full pb-20">
          <div className="mt-4">
            <div className="mb-8 flex justify-center">
              <img src="/safaricom-logo.png" alt="Safaricom Logo" className="h-24" />
            </div>
            
            {/* Voice Causal Analysis */}
            <div className="mb-6">
              <h3 className="font-semibold text-gray-700 mb-2 flex items-center">
                <BarChart2 className="h-4 w-4 mr-2 text-green-600" />
                <span className="text-green-700">Voice Causal Analysis</span>
              </h3>
              <div className="space-y-2">
                <div 
                  className="p-2 text-sm bg-green-50 rounded-md hover:bg-green-100 cursor-pointer"
                  onClick={() => handleExampleClick("What was the cause of the voice revenue change on 2024-03-01?")}
                >
                  What was the cause of the voice revenue change on 2024-03-01?
                </div>
                <div 
                  className="p-2 text-sm bg-green-50 rounded-md hover:bg-green-100 cursor-pointer"
                  onClick={() => handleExampleClick("Explain the voice revenue for the period 2024-03-01 to 2024-03-15")}
                >
                  Explain the voice revenue for the period 2024-03-01 to 2024-03-15
                </div>
                <div 
                  className="p-2 text-sm bg-green-50 rounded-md hover:bg-green-100 cursor-pointer"
                  onClick={() => handleExampleClick("Recommend actions for improving voice revenue")}
                >
                  Recommend actions for improving voice revenue
                </div>
              </div>
            </div>
            
            {/* Event Effects */}
            <div className="mb-6">
              <h3 className="font-semibold text-gray-700 mb-2 flex items-center">
                <TrendingUp className="h-4 w-4 mr-2 text-blue-600" />
                <span className="text-blue-700">Event Effects</span>
              </h3>
              <div className="space-y-2">
                <div 
                  className="p-2 text-sm bg-blue-50 rounded-md hover:bg-blue-100 cursor-pointer"
                  onClick={() => handleExampleClick("What was the effect of the new Tunukiwa campaign on 2024-02-15?")}
                >
                  What was the effect of the new Tunukiwa campaign on 2024-02-15?
                </div>
                <div 
                  className="p-2 text-sm bg-blue-50 rounded-md hover:bg-blue-100 cursor-pointer"
                  onClick={() => handleExampleClick("What were the effects on voice revenue on 2024-02-01, 2024-02-15, and 2024-03-01?")}
                >
                  What were the effects on voice revenue on 2024-02-01, 2024-02-15, and 2024-03-01?
                </div>
              </div>
            </div>
            
            {/* Forecasting */}
            <div className="mb-6">
              <h3 className="font-semibold text-gray-700 mb-2 flex items-center">
                <Calendar className="h-4 w-4 mr-2 text-purple-600" />
                <span className="text-purple-700">Forecasting</span>
              </h3>
              <div className="space-y-2">
                <div 
                  className="p-2 text-sm bg-purple-50 rounded-md hover:bg-purple-100 cursor-pointer"
                  onClick={() => handleExampleClick("Forecast the voice revenue for 2024-04-15")}
                >
                  Forecast the voice revenue for 2024-04-15
                </div>
                <div 
                  className="p-2 text-sm bg-purple-50 rounded-md hover:bg-purple-100 cursor-pointer"
                  onClick={() => handleExampleClick("Forecast from 2024-04-01 to 2024-04-30")}
                >
                  Forecast from 2024-04-01 to 2024-04-30
                </div>
              </div>
            </div>
            
            {/* Competition Analysis */}
            <div className="mb-6">
              <h3 className="font-semibold text-gray-700 mb-2 flex items-center">
                <Users className="h-4 w-4 mr-2 text-orange-600" />
                <span className="text-orange-700">Competition Analysis</span>
              </h3>
              <div className="space-y-2">
                <div 
                  className="p-2 text-sm bg-orange-50 rounded-md hover:bg-orange-100 cursor-pointer"
                  onClick={() => handleExampleClick("What's the comparison between Safaricom and competition?")}
                >
                  What's the comparison between Safaricom and competition?
                </div>
                <div 
                  className="p-2 text-sm bg-orange-50 rounded-md hover:bg-orange-100 cursor-pointer"
                  onClick={() => handleExampleClick("What are the Voice product views for Airtel online?")}
                >
                  What are the Voice product views for Airtel online?
                </div>
                <div 
                  className="p-2 text-sm bg-orange-50 rounded-md hover:bg-orange-100 cursor-pointer"
                  onClick={() => handleExampleClick("What negative sentiments on Safaricom exist online today?")}
                >
                  What negative sentiments on Safaricom exist online today?
                </div>
              </div>
            </div>
          </div>
          
          {dataRange && (
            <div className="mt-6">
              <h3 className="font-semibold text-gray-700 mb-2">Data Information</h3>
              <div className="bg-blue-50 p-3 rounded-md text-sm">
                <p><span className="font-medium">Date Range:</span></p>
                <p className="ml-2">{formatDate(dataRange.min_date)} to {formatDate(dataRange.max_date)}</p>
                <p><span className="font-medium">Total Days:</span> {dataRange.total_days}</p>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Main content */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-white to-green-50 border-b border-gray-200 p-4 flex items-center justify-between shadow-sm">
          <div className="flex items-center">
            <button 
              onClick={() => setSidebarOpen(!sidebarOpen)} 
              className="md:hidden mr-4 p-1 rounded hover:bg-gray-100"
            >
              <Menu className="h-6 w-6 text-gray-600" />
            </button>
            <div className="flex items-center space-x-3">
              <img src="/safaricom-logo.png" alt="Safaricom Logo" className="h-14" />
              <h1 className="text-xl font-bold text-green-800">Voice Causal Explainer</h1>
            </div>
          </div>
        </div>
        
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 bg-gradient-to-br from-green-50 via-gray-50 to-white">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            
            {isProcessing && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-md mr-12 flex items-center bg-opacity-95 backdrop-filter backdrop-blur-sm border-green-100">
                  <div className="h-8 w-8 rounded-full bg-green-100 flex items-center justify-center mr-3">
                    <Phone className="h-5 w-5 text-green-700" />
                  </div>
                  <LoadingIndicator />
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>
        
        {/* Input area */}
        <div className="bg-gradient-to-r from-green-50 to-white border-t border-green-100 border-opacity-50 p-4 shadow-inner">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className="relative">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask about voice revenue causality, effects, or forecasting..."
                className="w-full p-5 pr-14 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent shadow-lg text-base bg-white bg-opacity-95"
              />
              <button 
                type="submit"
                className="absolute right-3 top-1/2 transform -translate-y-1/2 p-3 rounded-lg bg-green-600 text-white hover:bg-green-700 focus:outline-none shadow-md"
              >
                <Send className="h-5 w-5" />
              </button>
            </div>
          </form>
          </div>
        </div>
    </div>
  );
};

export default VoiceCausalUI;
