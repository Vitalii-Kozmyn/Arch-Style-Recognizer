const analysis_btn = document.getElementById("analysis-btn");
const about_btn = document.getElementById("about-btn");

const analysis_section = document.querySelector(".upload-section");
const about_section = document.querySelector(".about-section");
const result_section = document.querySelector(".results-section");

analysis_btn.addEventListener("click", () => {
    analysis_btn.classList.add("active-option");
    analysis_btn.classList.remove("passive-option");

    about_btn.classList.add("passive-option");
    about_btn.classList.remove("active-option");

    analysis_section.classList.remove("hidden");
    about_section.classList.add("hidden");

    result_section.classList.add("hidden");
});

about_btn.addEventListener("click", () => {
    analysis_btn.classList.add("passive-option");
    analysis_btn.classList.remove("active-option");

    about_btn.classList.add("active-option");
    about_btn.classList.remove("passive-option");

    analysis_section.classList.add("hidden");
    about_section.classList.remove("hidden");

    result_section.classList.add("hidden");
});