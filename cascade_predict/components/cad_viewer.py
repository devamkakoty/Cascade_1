"""
3D CAD viewer component for Streamlit.

Renders STL files using Three.js and STEP files as point clouds.
Uses an iframe with postMessage to send geometry data to the viewer.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as st_components
import numpy as np

_VIEWER_HTML = Path(__file__).parent / "stl_viewer" / "index.html"


def render_cad_viewer(
    stl_bytes: bytes | None = None,
    vertices: np.ndarray | None = None,
    height: int = 500,
    key: str = "cad_viewer",
) -> None:
    """Render a 3D viewer in Streamlit.

    Args:
        stl_bytes: Raw STL file bytes (for full mesh rendering).
        vertices: Nx3 array of points (for STEP point cloud fallback).
        height: Viewer height in pixels.
        key: Streamlit widget key.
    """
    html_template = _VIEWER_HTML.read_text()

    if stl_bytes is not None:
        b64 = base64.b64encode(stl_bytes).decode("ascii")
        inject_script = f"""
        <script type="module">
        setTimeout(() => {{
            const iframe = document.querySelector('iframe');
            if (iframe) {{
                iframe.contentWindow.postMessage({{
                    type: 'stl_data',
                    payload: '{b64}'
                }}, '*');
            }}
        }}, 1000);
        </script>
        """
        # Inject the STL data directly into the HTML
        html_with_data = html_template.replace(
            "window.parent.postMessage({ type: 'viewer_ready' }, '*');",
            f"""
            window.parent.postMessage({{ type: 'viewer_ready' }}, '*');
            // Auto-load STL data
            const binary = atob('{b64}');
            const buffer = new ArrayBuffer(binary.length);
            const view = new Uint8Array(buffer);
            for (let i = 0; i < binary.length; i++) view[i] = binary.charCodeAt(i);
            loadSTLFromArrayBuffer(buffer);
            """,
        )
        st_components.html(html_with_data, height=height, scrolling=False)

    elif vertices is not None and len(vertices) > 0:
        # Subsample if too many points
        if len(vertices) > 50000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(vertices), 50000, replace=False)
            vertices = vertices[idx]

        pts_json = json.dumps(vertices.tolist())
        html_with_data = html_template.replace(
            "window.parent.postMessage({ type: 'viewer_ready' }, '*');",
            f"""
            window.parent.postMessage({{ type: 'viewer_ready' }}, '*');
            // Auto-load point cloud
            loadPointCloud({pts_json});
            """,
        )
        st_components.html(html_with_data, height=height, scrolling=False)
    else:
        st.info("No geometry data to display.")
