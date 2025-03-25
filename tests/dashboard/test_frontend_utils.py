"""
Tests for the Dashboard-specific UI components and interactions.

This module tests the modern dashboard's UI components including the notification
system, guided tour functionality, data visualization, and interactive controls.
"""

import pytest
import unittest.mock as mock
import json
from pathlib import Path

# Mock for DOM manipulation and JS execution
class MockDocument:
    """Mock implementation of document for testing."""
    
    def __init__(self):
        self.elements = {}
        self.event_handlers = {}
        self.local_storage = {}
    
    def getElementById(self, id):
        """Mock getElementById"""
        if id not in self.elements:
            self.elements[id] = MockElement(id)
        return self.elements[id]
    
    def getElementsByClassName(self, class_name):
        """Mock getElementsByClassName"""
        return [e for e in self.elements.values() if class_name in e.classList]
    
    def createElement(self, tag):
        """Mock createElement"""
        element = MockElement(f"dynamic-{tag}-{len(self.elements)}")
        element.tagName = tag
        return element
    
    def addEventListener(self, event, handler):
        """Mock addEventListener"""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    def dispatchEvent(self, event):
        """Mock dispatchEvent"""
        if event.type in self.event_handlers:
            for handler in self.event_handlers[event.type]:
                handler(event)
    
    def querySelectorAll(self, selector):
        """Mock querySelectorAll"""
        if selector == '[data-toggle="tooltip"]':
            return [e for e in self.elements.values() if e.getAttribute('data-toggle') == 'tooltip']
        return []


class MockElement:
    """Mock implementation of DOM element for testing."""
    
    def __init__(self, id):
        self.id = id
        self.innerHTML = ""
        self.textContent = ""
        self.value = ""
        self.style = {}
        self.attributes = {}
        self.classList = []
        self.dataset = {}
        self.children = []
        self.event_handlers = {}
        self.checked = False
        self.disabled = False
        self.tagName = "div"
    
    def getAttribute(self, name):
        """Mock getAttribute"""
        return self.attributes.get(name)
    
    def setAttribute(self, name, value):
        """Mock setAttribute"""
        self.attributes[name] = value
    
    def appendChild(self, child):
        """Mock appendChild"""
        self.children.append(child)
        return child
    
    def addEventListener(self, event, handler):
        """Mock addEventListener"""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    def dispatchEvent(self, event):
        """Mock dispatchEvent"""
        if event.type in self.event_handlers:
            for handler in self.event_handlers[event.type]:
                handler(event)
    
    def classList_add(self, class_name):
        """Mock classList.add"""
        if class_name not in self.classList:
            self.classList.append(class_name)
    
    def classList_remove(self, class_name):
        """Mock classList.remove"""
        if class_name in self.classList:
            self.classList.remove(class_name)
    
    def classList_toggle(self, class_name):
        """Mock classList.toggle"""
        if class_name in self.classList:
            self.classList.remove(class_name)
        else:
            self.classList.append(class_name)
        return class_name in self.classList


class MockEvent:
    """Mock implementation of DOM event for testing."""
    
    def __init__(self, type, target=None, data=None):
        self.type = type
        self.target = target
        self.data = data
        self.defaultPrevented = False
    
    def preventDefault(self):
        """Mock preventDefault"""
        self.defaultPrevented = True


class MockLocalStorage:
    """Mock implementation of localStorage for testing."""
    
    def __init__(self):
        self.items = {}
    
    def getItem(self, key):
        """Mock getItem"""
        return self.items.get(key)
    
    def setItem(self, key, value):
        """Mock setItem"""
        self.items[key] = value
    
    def removeItem(self, key):
        """Mock removeItem"""
        if key in self.items:
            del self.items[key]


@pytest.fixture
def mock_dom():
    """Create a mock DOM environment for testing."""
    return MockDocument()


@pytest.fixture
def mock_local_storage():
    """Create a mock localStorage for testing."""
    return MockLocalStorage()


@pytest.fixture
def mock_shepherd():
    """Create a mock Shepherd.js tour instance."""
    class MockTour:
        def __init__(self):
            self.steps = []
            self.started = False
        
        def addStep(self, step_id, options):
            self.steps.append({
                'id': step_id,
                'options': options
            })
        
        def start(self):
            self.started = True
        
        def cancel(self):
            self.started = False
        
        def next(self):
            pass
        
        def back(self):
            pass
    
    return MockTour()


class TestNotificationSystem:
    """Tests for the dashboard notification system."""
    
    def test_create_notification(self, mock_dom):
        """Test creating a notification."""
        # Mock document
        mock_document = mock_dom
        
        # Add notification container
        notification_center = MockElement("notification-center")
        mock_document.elements["notification-center"] = notification_center
        
        # Mock notification badge
        notification_badge = MockElement("notification-badge")
        mock_document.elements["notification-badge"] = notification_badge
        notification_badge.textContent = "0"
        
        # Mock notification count in localStorage
        mock_storage = {"dashboard_notifications": json.dumps([])}
        
        # Mock the createNotification function from JS
        def create_notification(title, message, type="info", auto_dismiss=True, persist=False):
            # Get existing notifications from localStorage
            notifications = json.loads(mock_storage.get("dashboard_notifications", "[]"))
            
            # Create new notification object
            notif = {
                "id": f"notif-{len(notifications) + 1}",
                "title": title,
                "message": message,
                "type": type,
                "timestamp": "2025-03-24T10:30:00",
                "read": False,
                "dismissed": False
            }
            
            # Add to persistent storage if needed
            if persist:
                notifications.append(notif)
                mock_storage["dashboard_notifications"] = json.dumps(notifications)
            
            # Create notification element
            notif_element = mock_document.createElement("div")
            notif_element.classList_add("notification")
            notif_element.classList_add(f"notification-{type}")
            notif_element.setAttribute("data-id", notif["id"])
            
            # Set content
            notif_element.innerHTML = f"""
                <div class="notification-header">
                    <span class="notification-title">{title}</span>
                    <button class="notification-close">×</button>
                </div>
                <div class="notification-body">{message}</div>
            """
            
            # Append to notification center
            notification_center.appendChild(notif_element)
            
            # Update badge count
            unread_count = sum(1 for n in notifications if not n.get("read", False))
            notification_badge.textContent = str(unread_count)
            notification_badge.classList_toggle("hidden", unread_count == 0)
            
            # Set auto-dismiss timeout
            if auto_dismiss:
                # In real JS we'd use setTimeout, here we'll just simulate dismissal
                self.dismiss_notification(notif["id"])
            
            return notif["id"]
        
        # Test creating a non-persistent notification
        notif_id = create_notification(
            "Test Notification", 
            "This is a test notification", 
            type="success", 
            auto_dismiss=False
        )
        
        # Verify notification element was created
        assert len(notification_center.children) == 1
        assert notification_center.children[0].classList == ["notification", "notification-success"]
        assert "Test Notification" in notification_center.children[0].innerHTML
        
        # Verify persistent storage was not updated (not persisted)
        notifications = json.loads(mock_storage.get("dashboard_notifications", "[]"))
        assert len(notifications) == 0
        
        # Test creating a persistent notification
        notif_id2 = create_notification(
            "Important Alert", 
            "This is a persistent notification", 
            type="error", 
            persist=True
        )
        
        # Verify element was created
        assert len(notification_center.children) == 2
        assert notification_center.children[1].classList == ["notification", "notification-error"]
        
        # Verify persistent storage was updated
        notifications = json.loads(mock_storage.get("dashboard_notifications", "[]"))
        assert len(notifications) == 1
        assert notifications[0]["title"] == "Important Alert"
        assert notifications[0]["type"] == "error"
        
        # Verify badge was updated
        assert notification_badge.textContent == "1"
        assert "hidden" not in notification_badge.classList
    
    def dismiss_notification(self, notif_id, mock_dom=None):
        """Helper method to dismiss a notification."""
        if not mock_dom:
            return
            
        # Find the notification element
        for child in mock_dom.elements["notification-center"].children:
            if child.getAttribute("data-id") == notif_id:
                # Remove from DOM
                mock_dom.elements["notification-center"].children.remove(child)
                break


class TestGuidedTour:
    """Tests for the guided tour functionality."""
    
    def test_create_tour(self, mock_dom, mock_shepherd):
        """Test creating a guided tour."""
        # Mock document
        mock_document = mock_dom
        
        # Create the tour steps (simulating JS implementation)
        def create_tour():
            # In real implementation, we'd create a Shepherd instance
            tour = mock_shepherd
            
            # Add welcome step
            tour.addStep('welcome', {
                'title': 'Welcome to the Dashboard',
                'text': 'This tour will guide you through the main features of the AI Trading dashboard.',
                'buttons': [
                    {'text': 'Skip', 'action': tour.cancel},
                    {'text': 'Next', 'action': tour.next}
                ]
            })
            
            # Add system status step
            tour.addStep('system-status', {
                'title': 'System Status',
                'text': 'This panel shows the current status of the trading system.',
                'attachTo': {'element': '#system-status-panel', 'on': 'bottom'},
                'buttons': [
                    {'text': 'Back', 'action': tour.back},
                    {'text': 'Next', 'action': tour.next}
                ]
            })
            
            # Add data visualization step
            tour.addStep('charts', {
                'title': 'Data Visualization',
                'text': 'These charts show real-time trading performance and market conditions.',
                'attachTo': {'element': '#charts-container', 'on': 'top'},
                'buttons': [
                    {'text': 'Back', 'action': tour.back},
                    {'text': 'Next', 'action': tour.next}
                ]
            })
            
            # Add controls step
            tour.addStep('controls', {
                'title': 'System Controls',
                'text': 'Use these controls to start/stop trading and adjust settings.',
                'attachTo': {'element': '#control-panel', 'on': 'left'},
                'buttons': [
                    {'text': 'Back', 'action': tour.back},
                    {'text': 'Finish', 'action': tour.next}
                ]
            })
            
            return tour
        
        # Create tour
        tour = create_tour()
        
        # Verify tour structure
        assert len(tour.steps) == 4
        assert tour.steps[0]['id'] == 'welcome'
        assert tour.steps[1]['id'] == 'system-status'
        assert tour.steps[2]['id'] == 'charts'
        assert tour.steps[3]['id'] == 'controls'
        
        # Verify step content
        assert 'Welcome to the Dashboard' in tour.steps[0]['options']['title']
        assert 'System Status' in tour.steps[1]['options']['title']
        assert 'Data Visualization' in tour.steps[2]['options']['title']
        
        # Start the tour
        tour.start()
        assert tour.started is True


class TestPerformanceOptimizations:
    """Tests for dashboard performance optimizations."""
    
    def test_lazy_loading_tabs(self, mock_dom):
        """Test lazy loading of tab content."""
        # Mock document elements
        mock_document = mock_dom
        
        # Add tab elements
        tab1 = MockElement("performance-tab")
        tab1.dataset["loaded"] = "false"
        tab1.dataset["contentUrl"] = "/api/performance_data"
        
        tab2 = MockElement("risk-tab")
        tab2.dataset["loaded"] = "false"
        tab2.dataset["contentUrl"] = "/api/risk_data"
        
        mock_document.elements["performance-tab"] = tab1
        mock_document.elements["risk-tab"] = tab2
        
        # Add loading indicator
        loading_indicator = MockElement("tab-loader")
        loading_indicator.classList = ["hidden"]
        mock_document.elements["tab-loader"] = loading_indicator
        
        # Mock API responses
        mock_api_responses = {
            "/api/performance_data": {
                "html": "<div class='performance-charts'>Charts content</div>",
                "data": {
                    "totalPnl": 15420.75,
                    "winRate": 68.5,
                    "trades": 285
                }
            },
            "/api/risk_data": {
                "html": "<div class='risk-metrics'>Risk content</div>",
                "data": {
                    "var": 2850.25,
                    "drawdown": 4.2,
                    "sharpe": 1.85
                }
            }
        }
        
        # Mock the lazyLoadTabContent function (implementation from JS)
        def lazy_load_tab_content(tab_id):
            tab = mock_document.getElementById(tab_id)
            
            # Show loading indicator
            loading_indicator.classList_remove("hidden")
            
            # Check if already loaded
            if tab.dataset["loaded"] == "true":
                # Hide loading indicator
                loading_indicator.classList_add("hidden")
                return
            
            # Get content URL from data attribute
            content_url = tab.dataset["contentUrl"]
            
            # For test, simulate API response
            response_data = mock_api_responses.get(content_url, {})
            
            # Update tab content
            if "html" in response_data:
                tab.innerHTML = response_data["html"]
            
            # Initialize any charts if needed (simulated)
            if "data" in response_data and tab_id == "performance-tab":
                # In real implementation would initialize charts
                tab.dataset["initialized"] = "true"
            
            # Mark as loaded
            tab.dataset["loaded"] = "true"
            
            # Hide loading indicator
            loading_indicator.classList_add("hidden")
            
            return response_data
        
        # Test loading performance tab
        result1 = lazy_load_tab_content("performance-tab")
        
        # Verify tab was loaded
        assert tab1.dataset["loaded"] == "true"
        assert tab1.innerHTML == "<div class='performance-charts'>Charts content</div>"
        assert tab1.dataset["initialized"] == "true"
        assert "hidden" in loading_indicator.classList
        assert result1["data"]["totalPnl"] == 15420.75
        
        # Test loading risk tab
        result2 = lazy_load_tab_content("risk-tab")
        
        # Verify tab was loaded
        assert tab2.dataset["loaded"] == "true"
        assert tab2.innerHTML == "<div class='risk-metrics'>Risk content</div>"
        assert "hidden" in loading_indicator.classList
        assert result2["data"]["var"] == 2850.25
    
    def test_chunked_rendering(self, mock_dom):
        """Test chunked rendering for large tables."""
        # Mock document
        mock_document = mock_dom
        
        # Create table element
        table_container = MockElement("trades-table-container")
        mock_document.elements["trades-table-container"] = table_container
        
        # Mock data (large dataset)
        large_data = [
            {"id": i, "symbol": f"BTC-USD", "price": 37500 + (i * 10), "size": 0.1 + (i * 0.01), "side": "buy" if i % 2 == 0 else "sell"}
            for i in range(1, 501)  # 500 rows
        ]
        
        # Mock the renderTableChunked function from JS
        def render_table_chunked(container_id, data, chunk_size=50, delay=10):
            container = mock_document.getElementById(container_id)
            
            # Create table structure
            table_html = """
            <table class="data-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Symbol</th>
                        <th>Price</th>
                        <th>Size</th>
                        <th>Side</th>
                    </tr>
                </thead>
                <tbody id="table-body"></tbody>
            </table>
            """
            container.innerHTML = table_html
            
            # Function to render a chunk (simplified for test)
            def render_chunk(start_idx):
                # Get the current chunk of data
                chunk = data[start_idx:start_idx + chunk_size]
                
                # Get the table body
                table_body = mock_document.createElement("tbody")
                
                # Add rows for this chunk
                for item in chunk:
                    row = mock_document.createElement("tr")
                    row.innerHTML = f"""
                        <td>{item['id']}</td>
                        <td>{item['symbol']}</td>
                        <td>{item['price']}</td>
                        <td>{item['size']}</td>
                        <td class="{item['side']}">{item['side']}</td>
                    """
                    table_body.appendChild(row)
                
                # Track rendered rows
                table_body.dataset["rows"] = str(len(chunk))
                
                # In real JS, we'd schedule next chunk with setTimeout
                # Here we'll just return the result
                return {
                    "chunk_index": start_idx // chunk_size,
                    "rows_rendered": len(chunk),
                    "tbody": table_body
                }
            
            # Start rendering from the first chunk
            # In real implementation, this would chain through setTimeout
            results = []
            for i in range(0, len(data), chunk_size):
                results.append(render_chunk(i))
            
            return results
        
        # Test chunked rendering
        chunks = render_table_chunked("trades-table-container", large_data)
        
        # Verify chunks
        assert len(chunks) == 10  # 500 items ÷ 50 per chunk = 10 chunks
        
        # Check first chunk
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["rows_rendered"] == 50
        assert len(chunks[0]["tbody"].children) == 50
        
        # Check last chunk
        assert chunks[9]["chunk_index"] == 9
        assert chunks[9]["rows_rendered"] == 50
        assert len(chunks[9]["tbody"].children) == 50


class TestPaginationSystem:
    """Tests for the pagination system."""
    
    def test_create_pagination(self, mock_dom):
        """Test creating pagination controls and using them."""
        # Mock document
        mock_document = mock_dom
        
        # Create pagination container
        pagination_container = MockElement("pagination-container")
        mock_document.elements["pagination-container"] = pagination_container
        
        # Create data container
        data_container = MockElement("data-container")
        mock_document.elements["data-container"] = data_container
        
        # Mock large dataset
        large_dataset = [f"Item {i}" for i in range(1, 201)]  # 200 items
        
        # Mock the createPagination function from JS
        def create_pagination(container_id, data_container_id, data, items_per_page=20):
            container = mock_document.getElementById(container_id)
            data_element = mock_document.getElementById(data_container_id)
            
            # Calculate number of pages
            total_pages = (len(data) + items_per_page - 1) // items_per_page
            
            # Create pagination controls
            pagination_html = "<div class='pagination-controls'>"
            pagination_html += "<button class='pagination-prev' disabled>Previous</button>"
            
            for i in range(1, total_pages + 1):
                active_class = " active" if i == 1 else ""
                pagination_html += f"<button class='pagination-page{active_class}' data-page='{i}'>{i}</button>"
            
            pagination_html += "<button class='pagination-next'>Next</button>"
            pagination_html += "</div>"
            
            container.innerHTML = pagination_html
            
            # Initialize with page 1
            update_page(1)
            
            # Function to update displayed data
            def update_page(page_num):
                # Calculate slice indices
                start_idx = (page_num - 1) * items_per_page
                end_idx = min(page_num * items_per_page, len(data))
                
                # Get items for current page
                current_items = data[start_idx:end_idx]
                
                # Update data container
                items_html = "<ul class='data-list'>"
                for item in current_items:
                    items_html += f"<li>{item}</li>"
                items_html += "</ul>"
                
                data_element.innerHTML = items_html
                
                # Update pagination buttons
                prev_button = container.getElementsByClassName("pagination-prev")[0]
                next_button = container.getElementsByClassName("pagination-next")[0]
                
                # Update disabled state
                prev_button.disabled = (page_num == 1)
                next_button.disabled = (page_num == total_pages)
                
                # Update active page
                for page_button in container.getElementsByClassName("pagination-page"):
                    if int(page_button.dataset["page"]) == page_num:
                        page_button.classList_add("active")
                    else:
                        page_button.classList_remove("active")
                
                return {
                    "current_page": page_num,
                    "total_pages": total_pages,
                    "items_shown": len(current_items)
                }
            
            # Simulate clicking on page 2
            page2_result = update_page(2)
            # Simulate clicking on Next
            page3_result = update_page(3)
            # Simulate clicking on Previous
            page2_again_result = update_page(2)
            
            return {
                "total_pages": total_pages,
                "page2_result": page2_result,
                "page3_result": page3_result,
                "page2_again_result": page2_again_result
            }
        
        # Test pagination
        result = create_pagination("pagination-container", "data-container", large_dataset)
        
        # Verify pagination
        assert result["total_pages"] == 10  # 200 items ÷ 20 per page = 10 pages
        
        # Check page 2 result
        assert result["page2_result"]["current_page"] == 2
        assert result["page2_result"]["items_shown"] == 20
        
        # Check page 3 result
        assert result["page3_result"]["current_page"] == 3
        assert result["page3_result"]["items_shown"] == 20
        
        # Check navigation back to page 2
        assert result["page2_again_result"]["current_page"] == 2