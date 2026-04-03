// Currency ticker and modal functionality
let latestRates = {};

function fetchRates() {
  fetch("https://open.er-api.com/v6/latest/ILS")
    .then((res) => res.json())
    .then((data) => {
      const rates = data.rates;

      latestRates = {
        "USD/ILS": 1 / rates.USD,
        "EUR/ILS": 1 / rates.EUR,
        "GBP/ILS": 1 / rates.GBP,
        "JOD/ILS": 1 / rates.JOD,
        "AED/ILS": 1 / rates.AED,
        "SAR/ILS": 1 / rates.SAR,
        "TRY/ILS": 1 / rates.TRY,
        "KWD/ILS": 1 / rates.KWD,
        "EGP/ILS": 1 / rates.EGP,
      };

      for (const key in latestRates) {
        const elId = key.split("/")[0].toLowerCase() + "-ils";
        const el = document.getElementById(elId);
        if (el) {
          el.textContent = latestRates[key].toFixed(2);
        }
      }
    })
    .catch((error) => {
      console.error("Failed to fetch currency rates:", error);
    });
}

function showCurrencyModal(pair) {
  const current = latestRates[pair];
  if (!current) return;

  document.getElementById("modal-title").textContent = pair;
  document.getElementById("modal-current").textContent = current.toFixed(2);
  document.getElementById("modal-high").textContent = (current + 0.05).toFixed(
    2,
  );
  document.getElementById("modal-low").textContent = (current - 0.05).toFixed(
    2,
  );
  document.getElementById("currencyModal").style.display = "block";
}

function closeCurrencyModal() {
  document.getElementById("currencyModal").style.display = "none";
}

// Initialize currency functionality
document.addEventListener("DOMContentLoaded", function () {
  // Modal close on outside click
  window.onclick = function (event) {
    const modal = document.getElementById("currencyModal");
    if (event.target === modal) {
      modal.style.display = "none";
    }
  };

  // Start fetching rates
  fetchRates();
  setInterval(fetchRates, 60000);
});
