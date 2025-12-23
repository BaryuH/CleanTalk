const TOXICITY_API_URL = "/api/v1/CleanTalk1";
// const TOXICITY_API_URL = "http://localhost:8000/api/v1/CleanTalk1";


const LABEL_IMAGES = {
  SAFE: "/images/safe.jpg",
  WARNING: "/images/warning.webp",
  BAN: "/images/ban.png"
};

class ThemeManager {
  constructor() {
    this.themeToggle = document.getElementById("themeToggle");
    this.isDarkMode = localStorage.getItem("theme") === "dark";

    this.init();
  }
  init() {
    this.applyTheme();
    if (this.themeToggle) {
      this.themeToggle.addEventListener("click", () => this.toggle());
    }
  }

  applyTheme() {
    if (this.isDarkMode) {
      document.body.classList.add("dark-mode");
    } else {
      document.body.classList.remove("dark-mode");
    }
  }

  toggle() {
    this.isDarkMode = !this.isDarkMode;
    localStorage.setItem("theme", this.isDarkMode ? "dark" : "light");
    this.applyTheme();
  }
}

class CharacterCounter {
  constructor() {
    this.textInput = document.getElementById("textInput");
    this.charCount = document.getElementById("charCount");

    if (this.textInput && this.charCount) {
      this.textInput.addEventListener("input", () => this.updateCount());
      this.updateCount();
    }
  }
  updateCount() {
    this.charCount.textContent = this.textInput.value.length;
  }
}

class ToxicityAnalyzer {
  constructor() {
    this.analyzeBtn = document.getElementById("analyzeBtn");
    this.textInput = document.getElementById("textInput");
    this.retryBtn = document.getElementById("retryBtn");
    this.emptyState = document.getElementById("emptyState");
    this.loadingState = document.getElementById("loadingState");
    this.errorState = document.getElementById("errorState");
    this.resultContent = document.getElementById("resultContent");
    this.errorMessage = document.getElementById("errorMessage");
    this.resultBadge = document.getElementById("resultBadge");
    this.badgeLabel = document.getElementById("badgeLabel");
    this.badgeDescription = document.getElementById("badgeDescription");
    this.issuesSection = document.getElementById("issuesSection");
    this.chipsContainer = document.getElementById("chipsContainer");
    this.noIssuesMessage = document.getElementById("noIssuesMessage");
    this.resultImageWrapper = document.getElementById("resultImageWrapper");
    this.resultImage = document.getElementById("resultImage");
    this.init();
  }

  init() {
    if (this.analyzeBtn) {
      this.analyzeBtn.addEventListener("click", () => this.analyze());
    }
    if (this.retryBtn) {
      this.retryBtn.addEventListener("click", () => this.analyze());
    }
  }

  async analyze() {
    const text = this.textInput.value.trim();

    if (!text) {
      alert("Please enter some text to analyze");
      return;
    }

    if (text.length < 3) {
      alert("Please enter at least 3 characters");
      return;
    }
    this.setLoading(true);
    try {
      const response = await this.callToxicityAPI(text);
      this.displayResult(response);
    } catch (error) {
      console.error("Error:", error);
      this.showError("Something went wrong while calling the API. Please try again.");
    } finally {
      this.setLoading(false);
    }
  }

  async callToxicityAPI(text) {
    const res = await fetch(TOXICITY_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }

    return await res.json();
  }

  getFinalLabelConfig(finalLabelRaw) {
    const label = (finalLabelRaw || "").toString().toUpperCase();
    if (label === "SAFE") {
      return { key: "SAFE", className: "safe", description: "This comment looks fine." };
    }
    if (label === "BAN") {
      return { key: "BAN", className: "ban", description: "This comment is highly toxic and should be blocked." };
    }
    return { key: "WARNING", className: "warning", description: "This comment may contain harmful or offensive elements." };
  }

  displayResult(data) {
    if (!data) {
      this.showError("Empty response from server.");
      return;
    }

    const config = this.getFinalLabelConfig(data.final_label);

    this.badgeLabel.textContent = config.key;

    this.resultBadge.className = "result-badge";
    this.resultBadge.classList.add(config.className);

    this.badgeDescription.textContent = config.description;

    if (this.resultImage && this.resultImageWrapper) {
      const imgURL = LABEL_IMAGES[config.key] || "";
      if (imgURL) {
        this.resultImage.src = imgURL;
        this.resultImage.alt = `${config.key} illustration`;
        this.resultImageWrapper.classList.remove("hidden");
      } else {
        this.resultImageWrapper.classList.add("hidden");
      }
    }

    const labelsObj = data.labels || {};
    const detectedLabels = Object.entries(labelsObj)
      .filter(([, v]) => v === 1 || v === true)
      .map(([key]) => key);

    this.chipsContainer.innerHTML = "";
    if (detectedLabels.length > 0) {
      this.issuesSection.style.display = "flex";
      this.noIssuesMessage.classList.add("hidden");
      detectedLabels.forEach(label => {
        const chip = document.createElement("div");
        chip.className = "chip";
        chip.innerHTML = `
          <span class="chip-dot"></span>
          <span>${label}</span>
        `;
        this.chipsContainer.appendChild(chip);
      });
    } else {
      this.issuesSection.style.display = "none";
      this.noIssuesMessage.classList.remove("hidden");
    }

    this.showResultContent();
  }

  showResultContent() {
    this.emptyState.classList.add("hidden");
    this.loadingState.classList.add("hidden");
    this.errorState.classList.add("hidden");
    this.resultContent.classList.remove("hidden");
  }

  setLoading(isLoading) {
    if (isLoading) {
      this.emptyState.classList.add("hidden");
      this.errorState.classList.add("hidden");
      this.resultContent.classList.add("hidden");
      this.loadingState.classList.remove("hidden");
    } else {
      this.loadingState.classList.add("hidden");
    }

    if (!this.analyzeBtn) return;
    this.analyzeBtn.disabled = isLoading;
    const btnText = this.analyzeBtn.querySelector(".btn-text");
    const btnLoader = this.analyzeBtn.querySelector(".btn-loader");
    if (btnText && btnLoader) {
      if (isLoading) {
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
      } else {
        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
      }
    }
  }

  showError(message) {
    this.emptyState.classList.add("hidden");
    this.loadingState.classList.add("hidden");
    this.resultContent.classList.add("hidden");
    this.errorState.classList.remove("hidden");
    if (this.errorMessage) {
      this.errorMessage.textContent = message;
    }
  }
}

function preloadImages(urls) {
  urls.forEach((url) => {
    const img = new Image();
    img.src = url;
  });
}

window.addEventListener("DOMContentLoaded", () => {
  preloadImages(Object.values(LABEL_IMAGES));
});

document.addEventListener("DOMContentLoaded", () => {
  new ThemeManager();
  new CharacterCounter();
  new ToxicityAnalyzer();

  document.querySelectorAll(".sample-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-text");
      const textInput = document.getElementById("textInput");
      textInput.value = text;
      textInput.dispatchEvent(new Event("input"));
    });
  });
});
