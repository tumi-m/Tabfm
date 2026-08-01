/* TabFM Lab console.
 *
 * Plain ES modules-free JavaScript on purpose: the UI ships as three static
 * files served by the API container, so there is no build step to run, no
 * toolchain to keep current, and nothing to fetch from a CDN at runtime.
 */
(function () {
  "use strict";

  var API = "";
  var CONVERSION = "ecommerce-conversion";
  var PAGE_VALUE = "ecommerce-page-value";
  var MATCH_RESULT = "sports-match-result";

  /* Curated e-commerce inputs. The models accept partial records and fill the
   * rest from training medians, so the form shows the fields a human can
   * reason about rather than all seventeen. */
  var ECOMMERCE_FIELDS = [
    { name: "PageValues", label: "Page value" },
    { name: "ProductRelated", label: "Product pages" },
    { name: "ProductRelated_Duration", label: "Time on product pages (s)" },
    { name: "ExitRates", label: "Exit rate" },
    { name: "BounceRates", label: "Bounce rate" },
    { name: "Administrative", label: "Account pages" },
    { name: "SpecialDay", label: "Special day proximity" },
    { name: "Month", label: "Month" },
    { name: "VisitorType", label: "Visitor type" },
    { name: "Weekend", label: "Weekend" }
  ];

  /* Outcome label and categorical slot. Slot order is fixed, never cycled. */
  var OUTCOMES = [
    { key: "H", label: "Home win", series: 1 },
    { key: "D", label: "Draw", series: 2 },
    { key: "A", label: "Away win", series: 3 }
  ];

  var state = { models: {}, details: {} };

  /* ------------------------------------------------------------- helpers */
  function $(id) { return document.getElementById(id); }

  function request(path, options) {
    return fetch(API + path, options).then(function (response) {
      return response.json().then(function (body) {
        if (!response.ok) {
          throw new Error(body && body.detail ? body.detail : "HTTP " + response.status);
        }
        return body;
      }).catch(function (error) {
        if (error instanceof SyntaxError) throw new Error("HTTP " + response.status);
        throw error;
      });
    });
  }

  function postJSON(path, payload) {
    return request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
  }

  function showError(message) {
    var toast = $("error-toast");
    toast.textContent = message;
    toast.hidden = false;
    clearTimeout(showError.timer);
    showError.timer = setTimeout(function () { toast.hidden = true; }, 6000);
  }

  function percent(value) {
    return value === null || value === undefined || isNaN(value)
      ? "—" : (value * 100).toFixed(1) + "%";
  }

  function number(value, digits) {
    return value === null || value === undefined || isNaN(value)
      ? "—" : Number(value).toFixed(digits === undefined ? 2 : digits);
  }

  function signed(value) {
    if (value === null || value === undefined || isNaN(value)) return "—";
    return (value >= 0 ? "+" : "") + (value * 100).toFixed(1) + "%";
  }

  /* --------------------------------------------------------------- setup */
  function loadModels() {
    return request("/api/models").then(function (models) {
      models.forEach(function (model) { state.models[model.task] = model; });
      renderModelTable(models);
      $("status-pill").textContent = models.length + " model" +
        (models.length === 1 ? "" : "s") + " loaded";
      $("status-pill").className = "pill " + (models.length ? "pill--ok" : "pill--bad");

      var wanted = [CONVERSION, PAGE_VALUE, MATCH_RESULT].filter(function (task) {
        return Boolean(state.models[task]);
      });
      return Promise.all(wanted.map(function (task) {
        return request("/api/models/" + task).then(function (detail) {
          state.details[task] = detail;
        });
      }));
    }).then(function () {
      buildEcommerceForm();
      buildTeamPickers();
    });
  }

  function renderModelTable(models) {
    var body = $("model-rows");
    body.innerHTML = "";
    if (!models.length) {
      body.innerHTML = '<tr><td colspan="7">No artifacts loaded. ' +
        "Run <code>tabfm-lab-train</code> and restart the API.</td></tr>";
      return;
    }
    models.forEach(function (model) {
      var keys = Object.keys(model.metrics || {}).slice(0, 3);
      var summary = keys.map(function (key) {
        return key + " " + number(model.metrics[key], 3);
      }).join(" · ");

      var row = document.createElement("tr");
      row.innerHTML =
        "<td>" + model.task + "</td>" +
        "<td>" + model.industry + "</td>" +
        "<td>" + model.task_type + "</td>" +
        "<td>" + model.model_name + "</td>" +
        '<td class="num">' + model.n_train.toLocaleString() + "</td>" +
        '<td class="num">' + model.n_test.toLocaleString() + "</td>" +
        "<td>" + (summary || "—") + "</td>";
      body.appendChild(row);
    });
  }

  /* --------------------------------------------------------- e-commerce */
  function fieldSpec(task, name) {
    var detail = state.details[task];
    if (!detail) return null;
    for (var i = 0; i < detail.fields.length; i += 1) {
      if (detail.fields[i].name === name) return detail.fields[i];
    }
    return null;
  }

  function buildEcommerceForm() {
    var container = $("ecommerce-fields");
    container.innerHTML = "";
    if (!state.details[CONVERSION]) {
      container.innerHTML = "<p>The conversion model is not loaded.</p>";
      return;
    }

    ECOMMERCE_FIELDS.forEach(function (field) {
      var spec = fieldSpec(CONVERSION, field.name);
      if (!spec) return;

      var label = document.createElement("label");
      label.className = "field" + (spec.dtype === "boolean" ? " field--check" : "");

      var caption = document.createElement("span");
      caption.className = "field__label";
      caption.textContent = field.label;

      var input;
      if (spec.choices && spec.choices.length) {
        input = document.createElement("select");
        spec.choices.forEach(function (choice) {
          var option = document.createElement("option");
          option.value = choice;
          option.textContent = choice;
          if (choice === spec.default) option.selected = true;
          input.appendChild(option);
        });
      } else if (spec.dtype === "boolean") {
        input = document.createElement("input");
        input.type = "checkbox";
        input.checked = Number(spec.default) >= 0.5;
      } else {
        input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.value = spec.default === null ? "" : Number(spec.default).toFixed(2);
      }
      input.id = "ec-" + field.name;
      input.dataset.feature = field.name;

      if (spec.dtype === "boolean") {
        label.appendChild(input);
        label.appendChild(caption);
      } else {
        label.appendChild(caption);
        label.appendChild(input);
      }
      container.appendChild(label);
    });
  }

  function collectEcommerceRecord() {
    var record = {};
    ECOMMERCE_FIELDS.forEach(function (field) {
      var input = $("ec-" + field.name);
      if (!input) return;
      if (input.type === "checkbox") {
        record[field.name] = input.checked ? 1 : 0;
      } else if (input.tagName === "SELECT") {
        record[field.name] = input.value;
      } else if (input.value !== "") {
        record[field.name] = Number(input.value);
      }
    });
    return record;
  }

  /* Each model was trained on its own column set — the page-value model has no
   * PageValues column, for instance — and the API rejects unknown features
   * rather than ignoring them, so narrow the record per model. */
  function restrict(record, task) {
    var detail = state.details[task];
    if (!detail) return record;
    var allowed = {};
    detail.feature_columns.forEach(function (name) { allowed[name] = true; });
    var narrowed = {};
    Object.keys(record).forEach(function (key) {
      if (allowed[key]) narrowed[key] = record[key];
    });
    return narrowed;
  }

  function scoreSession(event) {
    event.preventDefault();
    var button = event.target.querySelector("button[type=submit]");
    button.disabled = true;

    var record = collectEcommerceRecord();
    var calls = [postJSON("/api/models/" + CONVERSION + "/predict",
                          { records: [restrict(record, CONVERSION)] })];

    if (state.models[PAGE_VALUE]) {
      calls.push(postJSON("/api/models/" + PAGE_VALUE + "/predict",
                          { records: [restrict(record, PAGE_VALUE)] }));
    }

    Promise.all(calls).then(function (responses) {
      renderConversion(responses[0]);
      if (responses[1]) {
        var prediction = responses[1].predictions[0];
        var value = prediction.prediction_original_units;
        $("page-value").textContent = value === null || value === undefined
          ? "—" : number(value, 2);
      }
      $("ecommerce-results").hidden = false;
    }).catch(function (error) {
      showError(error.message);
    }).finally(function () {
      button.disabled = false;
    });
  }

  function renderConversion(response) {
    var prediction = response.predictions[0];
    var probabilities = prediction.probabilities || {};
    /* Classes come back as strings; the positive class is "1" (purchased). */
    var probability = probabilities["1"];
    if (probability === undefined) probability = probabilities["True"];

    $("conversion-value").textContent = percent(probability);
    $("conversion-bar").style.width = Math.max(0, Math.min(1, probability || 0)) * 100 + "%";
    $("ecommerce-latency").textContent = number(response.elapsed_ms, 1) + " ms";

    var base = 0.155; /* training-set conversion rate */
    var lift = probability ? probability / base : 0;
    $("conversion-context").textContent = probability === undefined ? "" :
      "The average session in the training data converts at " + percent(base) +
      ", so this is " + number(lift, 1) + "× the base rate.";
  }

  /* --------------------------------------------------------------- sports */
  function buildTeamPickers() {
    var detail = state.details[MATCH_RESULT];
    var home = $("home-team");
    var away = $("away-team");
    if (!detail || !detail.teams || !detail.teams.length) {
      [home, away].forEach(function (select) {
        select.innerHTML = "<option>No football model loaded</option>";
        select.disabled = true;
      });
      return;
    }

    detail.teams.forEach(function (team) {
      [home, away].forEach(function (select) {
        var option = document.createElement("option");
        option.value = team;
        option.textContent = team;
        select.appendChild(option);
      });
    });
    home.selectedIndex = 0;
    away.selectedIndex = Math.min(1, detail.teams.length - 1);
  }

  function collectOdds() {
    var home = parseFloat($("odds-home").value);
    var draw = parseFloat($("odds-draw").value);
    var away = parseFloat($("odds-away").value);
    if ([home, draw, away].some(function (value) { return isNaN(value) || value <= 1; })) {
      return null;
    }
    return { home: home, draw: draw, away: away };
  }

  function predictFixture(event) {
    event.preventDefault();
    var button = event.target.querySelector("button[type=submit]");

    var payload = {
      home_team: $("home-team").value,
      away_team: $("away-team").value
    };
    if (payload.home_team === payload.away_team) {
      showError("Pick two different teams.");
      return;
    }
    if ($("fixture-date").value) payload.date = $("fixture-date").value;
    var odds = collectOdds();
    if (odds) payload.odds = odds;

    button.disabled = true;
    postJSON("/api/fixtures/predict", payload).then(function (response) {
      renderFixture(response);
      $("fixture-results").hidden = false;
    }).catch(function (error) {
      showError(error.message);
    }).finally(function () {
      button.disabled = false;
    });
  }

  function renderFixture(response) {
    $("fixture-title").textContent =
      response.home_team + " vs " + response.away_team +
      " — " + String(response.date).slice(0, 10);

    var warning = $("reliability-warning");
    if (response.reliable && !(response.notes || []).length) {
      warning.hidden = true;
    } else {
      warning.hidden = false;
      $("reliability-text").textContent =
        (response.notes || []).join(" ") || "Features may be unreliable.";
    }

    var bars = $("result-bars");
    bars.innerHTML = "";
    var probabilities = response.result_probabilities || {};
    OUTCOMES.forEach(function (outcome) {
      var value = probabilities[outcome.key];
      if (value === undefined) return;

      var row = document.createElement("div");
      row.className = "bar-row";
      row.dataset.series = outcome.series;
      row.innerHTML =
        '<span class="bar-row__label">' + outcome.label + "</span>" +
        '<span class="bar-row__track">' +
          '<span class="bar-row__fill" style="width:' +
          (Math.max(0, Math.min(1, value)) * 100).toFixed(1) + '%"></span>' +
        "</span>" +
        '<span class="bar-row__value">' + percent(value) + "</span>";
      bars.appendChild(row);
    });

    $("expected-goals").textContent = number(response.expected_total_goals, 2);
    $("over-25").textContent = percent(response.over_2_5_probability);

    renderValueBets(response);
  }

  function renderValueBets(response) {
    var section = $("value-section");
    var rows = $("value-rows");
    rows.innerHTML = "";

    if (!response.value_bets || !response.value_bets.length) {
      section.hidden = true;
      return;
    }
    section.hidden = false;

    var byKey = {};
    OUTCOMES.forEach(function (outcome) { byKey[outcome.key] = outcome; });

    response.value_bets.forEach(function (bet) {
      var outcome = byKey[bet.outcome] || { label: bet.outcome, series: 1 };
      var row = document.createElement("tr");
      row.innerHTML =
        "<td><span class=\"swatch\" data-series=\"" + outcome.series +
          "\" aria-hidden=\"true\"></span>" + outcome.label + "</td>" +
        '<td class="num">' + percent(bet.model_probability) + "</td>" +
        '<td class="num">' + percent(bet.market_probability) + "</td>" +
        '<td class="num">' + number(bet.decimal_odds, 2) + "</td>" +
        '<td class="num">' + signed(bet.edge) + "</td>" +
        "<td class=\"verdict " + (bet.recommended ? "verdict--value" : "verdict--none") +
          "\">" + (bet.recommended ? "✔ Value" : "— No bet") + "</td>";
      rows.appendChild(row);
    });

    $("overround-note").textContent = response.market_overround
      ? "Bookmaker overround on these prices: " +
        ((response.market_overround - 1) * 100).toFixed(1) + "%."
      : "";
  }

  /* ----------------------------------------------------------------- chrome */
  function initTabs() {
    var tabs = document.querySelectorAll(".tab");
    Array.prototype.forEach.call(tabs, function (tab) {
      tab.addEventListener("click", function () {
        Array.prototype.forEach.call(tabs, function (other) {
          other.setAttribute("aria-selected", String(other === tab));
          $("panel-" + other.dataset.panel).hidden = other !== tab;
        });
      });
    });
  }

  function initTheme() {
    $("theme-toggle").addEventListener("click", function () {
      var root = document.documentElement;
      var dark = root.getAttribute("data-theme") === "dark" ||
        (!root.getAttribute("data-theme") &&
          window.matchMedia("(prefers-color-scheme: dark)").matches);
      root.setAttribute("data-theme", dark ? "light" : "dark");
    });
  }

  function init() {
    initTabs();
    initTheme();
    $("ecommerce-form").addEventListener("submit", scoreSession);
    $("fixture-form").addEventListener("submit", predictFixture);
    document.querySelector("[data-action=reset-ecommerce]")
      .addEventListener("click", buildEcommerceForm);

    loadModels().catch(function (error) {
      $("status-pill").textContent = "API unreachable";
      $("status-pill").className = "pill pill--bad";
      showError("Could not reach the API: " + error.message);
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
