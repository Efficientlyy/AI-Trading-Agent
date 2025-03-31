/**
 * Progress Circle JavaScript
 * 
 * Initializes and updates progress circles in the monitoring dashboard.
 */

// Initialize progress circles
function initProgressCircles() {
    const progressCircles = document.querySelectorAll('.progress-circle');

    progressCircles.forEach(circle => {
        const value = parseInt(circle.getAttribute('data-value')) || 0;
        const circumference = 283; // 2 * PI * 45 (radius)
        const offset = circumference - (value / 100) * circumference;

        // Create SVG
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        const circleEl = document.createElementNS('http://www.w3.org/2000/svg', 'circle');

        // Set attributes
        svg.setAttribute('viewBox', '0 0 100 100');
        circleEl.setAttribute('cx', '50');
        circleEl.setAttribute('cy', '50');
        circleEl.setAttribute('r', '45');
        circleEl.style.strokeDashoffset = offset;

        // Append elements
        svg.appendChild(circleEl);
        circle.appendChild(svg);

        // Set color class
        if (value >= 80) {
            circle.classList.add('danger');
        } else if (value >= 50) {
            circle.classList.add('warning');
        } else {
            circle.classList.add('success');
        }
    });
}

// Update progress circle
function updateProgressCircle(circle, value) {
    // Get or create SVG and circle elements
    let svg = circle.querySelector('svg');
    let circleEl = svg ? svg.querySelector('circle') : null;

    if (!svg || !circleEl) {
        // Create elements if they don't exist
        svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        circleEl = document.createElementNS('http://www.w3.org/2000/svg', 'circle');

        // Set attributes
        svg.setAttribute('viewBox', '0 0 100 100');
        circleEl.setAttribute('cx', '50');
        circleEl.setAttribute('cy', '50');
        circleEl.setAttribute('r', '45');

        // Append elements
        svg.appendChild(circleEl);
        circle.appendChild(svg);
    }

    // Update value
    const circumference = 283; // 2 * PI * 45 (radius)
    const offset = circumference - (value / 100) * circumference;
    circleEl.style.strokeDashoffset = offset;

    // Update text
    const valueEl = circle.querySelector('.progress-circle-value');
    if (valueEl) {
        valueEl.textContent = `${value}%`;
    }

    // Update color class
    circle.classList.remove('success', 'warning', 'danger');
    if (value >= 80) {
        circle.classList.add('danger');
    } else if (value >= 50) {
        circle.classList.add('warning');
    } else {
        circle.classList.add('success');
    }

    // Update data attribute
    circle.setAttribute('data-value', value);
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function () {
    initProgressCircles();
});