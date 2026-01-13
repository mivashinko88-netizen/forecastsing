// charts.js - Plotly.js Chart Configurations

// Chart theme matching our design system
const chartTheme = {
    colors: ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316'],
    font: {
        family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        size: 12,
        color: '#374151'
    },
    gridColor: '#E5E7EB',
    backgroundColor: 'white'
};

// Create forecast line chart
function createForecastChart(containerId, predictions, options = {}) {
    if (!predictions || predictions.length === 0) {
        return;
    }

    // Group predictions by item
    const itemData = {};
    predictions.forEach(p => {
        const itemName = p.item_name || 'All Items';
        if (!itemData[itemName]) {
            itemData[itemName] = { dates: [], values: [] };
        }
        itemData[itemName].dates.push(p.date);
        itemData[itemName].values.push(p.predicted_quantity);
    });

    // Create traces for each item
    const traces = Object.entries(itemData).map(([name, data], index) => ({
        x: data.dates,
        y: data.values,
        type: 'scatter',
        mode: 'lines+markers',
        name: name,
        line: {
            width: 2,
            shape: 'spline',
            color: chartTheme.colors[index % chartTheme.colors.length]
        },
        marker: {
            size: 6,
            color: chartTheme.colors[index % chartTheme.colors.length]
        },
        hovertemplate: '%{x|%b %d}<br>%{y} units<extra>%{fullData.name}</extra>'
    }));

    const layout = {
        font: chartTheme.font,
        paper_bgcolor: chartTheme.backgroundColor,
        plot_bgcolor: chartTheme.backgroundColor,
        margin: { t: 20, r: 20, b: 50, l: 50 },
        xaxis: {
            gridcolor: chartTheme.gridColor,
            linecolor: chartTheme.gridColor,
            tickformat: '%b %d',
            tickfont: { size: 11, color: '#6B7280' },
            showgrid: true,
            zeroline: false
        },
        yaxis: {
            gridcolor: chartTheme.gridColor,
            linecolor: chartTheme.gridColor,
            tickfont: { size: 11, color: '#6B7280' },
            title: options.yAxisTitle || 'Predicted Quantity',
            titlefont: { size: 12, color: '#6B7280' },
            showgrid: true,
            zeroline: false
        },
        legend: {
            orientation: 'h',
            y: -0.15,
            x: 0.5,
            xanchor: 'center',
            font: { size: 11 }
        },
        hoverlabel: {
            bgcolor: '#1F2937',
            bordercolor: '#1F2937',
            font: { color: 'white', size: 12 }
        },
        hovermode: 'x unified',
        ...options.layout
    };

    const config = {
        responsive: true,
        displayModeBar: false,
        displaylogo: false
    };

    Plotly.newPlot(containerId, traces, layout, config);
}

// Create feature importance horizontal bar chart
function createFeatureImportanceChart(containerId, features, options = {}) {
    if (!features || Object.keys(features).length === 0) {
        return;
    }

    // Sort features by importance and take top 10
    const sortedFeatures = Object.entries(features)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);

    const trace = {
        y: sortedFeatures.map(f => formatFeatureName(f[0])),
        x: sortedFeatures.map(f => (f[1] * 100).toFixed(1)),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: chartTheme.colors[0],
            line: { width: 0 }
        },
        hovertemplate: '%{y}: %{x}%<extra></extra>'
    };

    const layout = {
        font: chartTheme.font,
        paper_bgcolor: chartTheme.backgroundColor,
        plot_bgcolor: chartTheme.backgroundColor,
        margin: { t: 20, r: 30, b: 40, l: 120 },
        xaxis: {
            gridcolor: chartTheme.gridColor,
            ticksuffix: '%',
            tickfont: { size: 11, color: '#6B7280' },
            title: 'Importance',
            titlefont: { size: 12, color: '#6B7280' }
        },
        yaxis: {
            tickfont: { size: 11, color: '#374151' },
            automargin: true
        },
        hoverlabel: {
            bgcolor: '#1F2937',
            font: { color: 'white' }
        },
        ...options.layout
    };

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.newPlot(containerId, [trace], layout, config);
}

// Create stats summary donut chart
function createDonutChart(containerId, data, options = {}) {
    const trace = {
        values: data.map(d => d.value),
        labels: data.map(d => d.label),
        type: 'pie',
        hole: 0.6,
        marker: {
            colors: chartTheme.colors
        },
        textinfo: 'percent',
        textposition: 'outside',
        hovertemplate: '%{label}<br>%{value} (%{percent})<extra></extra>'
    };

    const layout = {
        font: chartTheme.font,
        paper_bgcolor: chartTheme.backgroundColor,
        margin: { t: 30, r: 30, b: 30, l: 30 },
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.1,
            x: 0.5,
            xanchor: 'center'
        },
        ...options.layout
    };

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.newPlot(containerId, [trace], layout, config);
}

// Helper function to format feature names
function formatFeatureName(name) {
    const nameMap = {
        'is_payday': 'Payday',
        'item_encoded': 'Item Type',
        'day_of_week': 'Day of Week',
        'is_holiday': 'Holiday',
        'is_school_break': 'School Break',
        'temp_max': 'Max Temperature',
        'temp_min': 'Min Temperature',
        'weather_code': 'Weather',
        'has_sports': 'Sports Event',
        'week_of_year': 'Week of Year',
        'precipitation': 'Precipitation',
        'day_of_month': 'Day of Month',
        'month': 'Month',
        'is_weekend': 'Weekend'
    };

    return nameMap[name] || name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// Update chart with new data
function updateChart(containerId, newData) {
    Plotly.react(containerId, newData.traces, newData.layout);
}

// Resize chart to fit container
function resizeChart(containerId) {
    Plotly.Plots.resize(document.getElementById(containerId));
}

// Export functions
window.createForecastChart = createForecastChart;
window.createFeatureImportanceChart = createFeatureImportanceChart;
window.createDonutChart = createDonutChart;
window.updateChart = updateChart;
window.resizeChart = resizeChart;
