document.getElementById("loginForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent default form submission

    // Get username and password from input fields
    var username = document.getElementById("username").value;
    var password = document.getElementById("password").value;

    // Perform login validation (replace with your actual validation logic)
    if (username === "admin" && password === "admin123") {
        // Redirect to another page upon successful login
        window.location.href = "home1.html";
    } else {
        alert("Invalid username or password. Please try again.");
    }
});