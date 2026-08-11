"""The graph's full-window control, pinned at the level the browser cares about.

There is no DOM here, so these assert the three things that actually break when
this feature regresses: the toggle targets an element that really wraps the
canvas, exactly one of the two buttons is reachable in each state, and the resize
event is replayed — without that last one the section grows and the graph keeps
rendering at its old size, which looks like a broken canvas rather than a missing
line of script.
"""

from __future__ import annotations

import re

from app.web import HTML


def _ontology_scene() -> str:
    match = re.search(r'<section class="ontology-scene[^"]*">(.*?)</section>', HTML, re.S)
    assert match, "the ontology scene section is gone"
    return match.group(1)


def test_both_buttons_live_inside_the_section_the_toggle_resizes() -> None:
    scene = _ontology_scene()
    assert 'id="enterOntologyFullscreen"' in scene
    assert 'id="exitOntologyFullscreen"' in scene
    # The canvas must be inside the same section, or making the section
    # full-window would leave the graph behind at its original size.
    assert 'id="ontologyCanvas"' in scene


def test_exactly_one_button_is_visible_in_each_state() -> None:
    assert "#exitOntologyFullscreen { display: none; }" in HTML
    assert ".ontology-scene.is-fullscreen #enterOntologyFullscreen { display: none; }" in HTML
    assert ".ontology-scene.is-fullscreen #exitOntologyFullscreen { display: inline-flex" in HTML


def test_fullscreen_rule_outranks_the_viewport_canvas_heights() -> None:
    """``#ontologyCanvas`` is re-height-ed in two media queries.

    A media query adds no specificity, so the full-window rule has to win on its
    own selector weight. ``.ontology-scene.is-fullscreen #ontologyCanvas`` is two
    classes plus an id against a bare id, which it does — but only while it keeps
    both classes, so the exact selector is what is pinned here.
    """
    assert ".ontology-scene.is-fullscreen #ontologyCanvas { height: 100vh; }" in HTML
    assert "#ontologyCanvas { height: 700px; }" in HTML  # the rule being outranked
    assert "#ontologyCanvas { height: 560px; }" in HTML


def test_the_toggle_replays_a_resize_so_the_renderer_follows() -> None:
    assert "function setOntologyFullscreen(" in HTML
    assert "window.dispatchEvent(new Event('resize'))" in HTML
    # Two nested frames: one for the class change to lay out, one so the new
    # canvas rect is readable when the renderer measures it.
    assert HTML.count("requestAnimationFrame(") >= 2


def test_escape_leaves_fullscreen() -> None:
    assert "event.key !== 'Escape'" in HTML
    assert "setupOntologyFullscreen();" in HTML, "the handlers are never wired up"


def test_page_scroll_is_locked_while_the_graph_owns_the_window() -> None:
    assert "body.ontology-fullscreen { overflow: hidden; }" in HTML
    assert "document.body.classList.toggle('ontology-fullscreen', on);" in HTML
