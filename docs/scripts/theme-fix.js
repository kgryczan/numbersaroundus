document.addEventListener("DOMContentLoaded", function () {
    const theme = localStorage.getItem("quarto-theme");
    if (theme === "dark") {
        document.documentElement.setAttribute("data-theme", "dark");
    }
});
