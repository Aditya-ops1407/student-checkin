(function () {
  "use strict";
  document.documentElement.classList.add("js");

  // Mobile menu
  var mb = document.querySelector(".menu-btn"), nav = document.getElementById("site-nav");
  if (mb && nav) {
    mb.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      mb.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  // Print button on result page
  var pb = document.getElementById("btn-print");
  if (pb) pb.addEventListener("click", function () { window.print(); });

  // Stepped check-in form
  var form = document.getElementById("checkin-form");
  if (!form) return;
  var steps = Array.prototype.slice.call(form.querySelectorAll(".step"));
  var crumbs = Array.prototype.slice.call(document.querySelectorAll(".progress li"));
  var next = document.getElementById("btn-next"), back = document.getElementById("btn-back"),
      submit = document.getElementById("btn-submit");
  var cur = 0;

  function show(i, focus) {
    cur = i;
    steps.forEach(function (s, n) { s.classList.toggle("active", n === i); });
    crumbs.forEach(function (c, n) {
      c.classList.toggle("current", n === i);
      c.classList.toggle("done", n < i);
      if (n === i) c.setAttribute("aria-current", "step"); else c.removeAttribute("aria-current");
    });
    var last = i === steps.length - 1;
    back.hidden = i === 0;
    next.style.display = last ? "none" : "";
    submit.classList.toggle("show", last);
    if (focus) {
      var h = steps[i].querySelector("h2");
      if (h) { h.setAttribute("tabindex", "-1"); h.focus(); }
      window.scrollTo({ top: 0 });
    }
  }

  function groupValid(step) {
    var ok = true, firstBad = null;
    var fields = step.querySelectorAll(".field");
    Array.prototype.forEach.call(fields, function (f) {
      var inputs = f.querySelectorAll("input");
      var valid = true, msg = "";
      if (inputs[0].type === "radio") {
        valid = Array.prototype.some.call(inputs, function (r) { return r.checked; });
        msg = "Choose one of the options.";
      } else {
        var el = inputs[0];
        valid = el.value !== "" && el.checkValidity();
        msg = el.value === "" ? "Enter a number." :
          "Enter a value between " + parseFloat(el.min) + " and " + parseFloat(el.max) + ".";
      }
      var old = f.querySelector(".err.js-err");
      if (old) old.remove();
      f.classList.toggle("invalid", !valid);
      if (!valid) {
        ok = false;
        var p = document.createElement("p");
        p.className = "err js-err"; p.setAttribute("role", "alert"); p.textContent = msg;
        f.appendChild(p);
        if (!firstBad) firstBad = f;
      }
    });
    if (firstBad) {
      var t = firstBad.querySelector("input"); if (t) t.focus();
    }
    return ok;
  }

  next.addEventListener("click", function () {
    if (groupValid(steps[cur])) show(cur + 1, true);
  });
  back.addEventListener("click", function () { show(cur - 1, true); });
  form.addEventListener("submit", function (e) {
    for (var i = 0; i < steps.length; i++) {
      if (!groupValid(steps[i])) { e.preventDefault(); show(i, false); return; }
    }
  });

  // After a server-side error, open the step that holds the first problem.
  var firstErr = form.querySelector(".err");
  var start = 0;
  if (firstErr) {
    var st = firstErr.closest(".step");
    if (st) start = steps.indexOf(st);
  }
  show(start, false);
})();
