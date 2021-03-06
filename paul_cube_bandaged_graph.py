"""
  This software draws bandage cube graphs

  by Nasca Octavian Paul
  http://www.paulnasca.com


  This software requires the bce library from:
  https://github.com/ladislavdubravsky/bandaged-cube-explorer
"""

import os
import sys
import csv
import re
import pydot
import numpy as np
import tempfile
import itertools
import collections
import networkx as nx
import glob
import shutil
import argparse
import svgwrite

import bce.core


"""
The configuration of the graph output
"""
MAX_NUMBER_OF_NODES_PER_CATEGORY = {
    "cube": 10,
    "circle_with_label": 50,
    "label_only": 0,
    "circle": 2500,
}
SHOW_EDGE_LABELS_MAX = 300
SHOW_EDGE_ARROWS_MAX = 2000

"""
face color scheme (it is also used to draw the face moves in the graph)
"""
FACE_COLORS = {
    "L": "green",
    "F": "red",
    "R": "blue",
    "B": "orange",
    "U": "grey",
    "D": "#E0E000",
}
UNKNOWN_FACE_EDGE_COLOR = "#8020a0"

# TODO: add better passing of the configuration from the command line
# to the lower level functions (like draw_svg_cube for the "cube_draw_projection")


def explore_cube(cube):
    """
    A wrapper to bce.explore of the cube

    Args:
       cube: the bandaged cube in array[27] format
    Returns:
       tuples of (nodes, edges, labels, i2c)
       
    """
    nodes, edges, labels, i2c, c2i = bce.core.explore(
        bce.core.normalize(cube), fullperm=False
    )
    result = nodes, edges, labels, i2c
    return result


def draw_svg_cube(
    svg_filename,
    cube_order,
    cube,
    image_size,
    color_mode="white",
    label=None,
    cube_draw_projection="cube_map",
):
    """
    Draw a bandaged NxNxN cube in svg format.

    Args:
         svg_filename: output file path of the image
         cube_order: the cube order (example: 3 for a 3x3x3 cube)
         cube: the bandaged cube in array[cube_order * cube_order * cube_order]
         image_size: output cube image size in units
         color_mode: how the colors are drawn
             The color_mode can be:
                "white"  - the cube is completely white
                "center" - only the center (and the cubies connected to the centers) are coloured
                "full"   - the whole faces are drawn with the face color
         label: cube label
         cube_draw_projection: the projection of the cube drawing ("cube_map", "isometric")
    """
    global FACE_COLORS
    CUBE_MODES = {
        "cube_map": {
            "transforms": {
                "L": " scale(0.25) translate(0,1)",
                "F": " scale(0.25) translate(1,1)",
                "R": " scale(0.25) translate(2,1)",
                "B": " scale(0.25) translate(3,1)",
                "U": " scale(0.25) translate(1,0)",
                "D": " scale(0.25) translate(1,2)",
            },
            "label_pos_x": 1.0,
            "label_pos_y": 0.75,
            "label_size": 0.17,
            "label_anchor": "end",
            "image_size_x_multiplier": 1.0,
            "image_size_y_multiplier": 0.75,
        },
        "isometric": {
            "transforms": {
                "U": "scale(0.3) translate(0.8,0) scale(1,0.58) rotate(45) rotate(-90,0.5,0.5)",
                "L": "scale(0.3) translate(0.8,0.82) rotate(60) scale(1,0.58) rotate(45) rotate(-90,0.5,0.5)",
                "F": "scale(0.3) translate(0.8,0.82) rotate(-60) scale(1,0.58) rotate(45)",
                "D": "scale(0.3) translate(2.41,0.82) scale(1,0.58) rotate(45) rotate(180,0.5,0.5)",
                "R": "scale(0.3) translate(2.41,0.82) rotate(120) scale(1,0.58) rotate(45) rotate(180,0.5,0.5)",
                "B": "scale(0.3) translate(2.41,0.82) rotate(-120) scale(1,0.58) rotate(45) rotate(90,0.5,0.5)",
            },
            "label_pos_x": 0.5,
            "label_pos_y": 0.63,
            "label_size": 0.17,
            "label_anchor": "middle",
            "image_size_x_multiplier": 1.0,
            "image_size_y_multiplier": 0.65,
        },
    }
    current_projection = CUBE_MODES[cube_draw_projection]
    dwg = svgwrite.Drawing(svg_filename, profile="tiny")

    stroke_width = 0.05 * cube_order
    g1 = svgwrite.container.Group(
        transform="scale({}) translate({})".format(
            image_size, stroke_width * 0.25 / cube_order
        )
    )
    cube_np = np.reshape(cube, (cube_order, cube_order, cube_order))

    faces = {
        "L": cube_np[:, :, 0],
        "F": cube_np[:, -1, :],
        "R": cube_np[:, ::-1, -1],
        "B": cube_np[:, 0, ::-1],
        "U": cube_np[0, :, :],
        "D": cube_np[-1, ::-1, :],
    }
    # draw the faces
    for face_name, face in faces.items():
        face = faces[face_name]
        transforms = current_projection["transforms"]
        grp = svgwrite.container.Group(
            transform=transforms[face_name] + " scale({})".format(1.0 / cube_order)
        )

        # draw full face center
        if color_mode == "full":
            grp.add(
                dwg.rect(
                    (0, 0),
                    (cube_order, cube_order),
                    fill=FACE_COLORS[face_name],
                    stroke_width=stroke_width / 2.0,
                    stroke=FACE_COLORS[face_name],
                )
            )

        # draw face center colors for odd size cubes
        if cube_order % 2 == 1 and color_mode == "center":
            center_pos = (cube_order - 1) // 2
            center_group = face[center_pos][center_pos]
        else:
            center_group = -1
            center_pos = -1
        for y in range(0, cube_order):
            for x in range(0, cube_order):
                if (center_group != 0 and face[y][x] == center_group) or (
                    x == center_pos and y == center_pos
                ):
                    grp.add(
                        dwg.rect(
                            (x, y),
                            (1, 1),
                            fill=FACE_COLORS[face_name],
                            stroke_width=stroke_width / 2.0,
                            stroke=FACE_COLORS[face_name],
                        )
                    )

        # draw the square separations (for non bandage squares)
        for y in range(0, cube_order):
            for x in range(0, cube_order):
                if (x < (cube_order - 1)) and (
                    face[y][x + 1] == 0 or (face[y][x + 1] != face[y][x])
                ):
                    grp.add(
                        dwg.line(
                            (x + 1, y),
                            (x + 1, y + 1),
                            stroke="black",
                            stroke_width=stroke_width,
                        )
                    )
                if (y < (cube_order - 1)) and (
                    face[y + 1][x] == 0 or (face[y + 1][x] != face[y][x])
                ):
                    grp.add(
                        dwg.line(
                            (x, y + 1),
                            (x + 1, y + 1),
                            stroke="black",
                            stroke_width=stroke_width,
                        )
                    )

        # draw face edges
        grp.add(
            dwg.rect(
                (0, 0),
                (cube_order, cube_order),
                fill="none",
                stroke="black",
                stroke_width=stroke_width * 1.5,
            )
        )

        g1.add(grp)

    dwg.add(g1)

    # draw cube label
    if label:
        label_pos_x = current_projection["label_pos_x"]
        label_pos_y = current_projection["label_pos_y"]
        label_anchor = current_projection["label_anchor"]
        label_size = current_projection["label_size"]

        dwg.add(
            dwg.text(
                label,
                (image_size * label_pos_x, image_size * label_pos_y),
                font_size=image_size * label_size,
                text_anchor=label_anchor,
                font_family="sans-serif",
                font_weight="bold",
            )
        )

    label_pos_x = current_projection["label_pos_x"]
    label_pos_y = current_projection["label_pos_y"]
    dwg["height"] = str(image_size * current_projection["image_size_y_multiplier"])
    dwg["width"] = str(image_size * current_projection["image_size_x_multiplier"])
    dwg.save(pretty=True)


def generate_hex_signature_connections():
    """
    Generates the connection representing every bit of the hex signatures
    for the 3x3x3 cube
    Returns:
        An array[54] of pairs of size 3 tuples of the x,y,z coordinates of the cubies
        The pair means that a connection is between both elements.
        Example:
        [..., ((1,2,1),(1,2,2)),...] means that the (1,2,1) and (1,2,2) cubies are connected
    """
    conn = []

    for z in range(3):
        conn += [((z, 0, 0), (z, 0, 1)), ((z, 0, 1), (z, 0, 2))]
        for y in range(2):
            conn += [
                ((z, y, 0), (z, y + 1, 0)),
                ((z, y, 1), (z, y + 1, 1)),
                ((z, y, 2), (z, y + 1, 2)),
                ((z, y + 1, 0), (z, y + 1, 1)),
                ((z, y + 1, 1), (z, y + 1, 2)),
            ]
        if z < 2:
            conn += [((z, m // 3, m % 3), (z + 1, m // 3, m % 3)) for m in range(9)]
    return conn


def convert_hex_signature_to_bandage_array(signature_hex):
    """
      Convert hex signature to an array[27] bandage cube
      
      Args:
          signature_hex: the hex signature of the cube

      Returns:
          a list of size[27] of the cubies  

      Example: 
          for signature_hex="33EC01800846" it returns
          [ 3, 4, 5, 
           6, 1, 2, 
          7, 1, 2, 
            3, 4, 5, 
           6, 1, 8, 
          7, 1, 8, 
            9, 9, 0, 
           10,10,11, 
          12,12,11]
    """
    cube_int = int(signature_hex, 16)
    CUBE_N_CONNECTIONS = 54

    signature_text_bin = format(cube_int, "0{}b".format(CUBE_N_CONNECTIONS))
    signature_bin = [m == "1" for m in signature_text_bin]
    if len(signature_bin) > CUBE_N_CONNECTIONS:
        raise ValueError("Invalid signature - too large")

    connections = [
        m
        for k, m in enumerate(generate_hex_signature_connections())
        if signature_bin[k]
    ]

    connections_graph = nx.Graph(connections)
    subgraphs = (connections_graph.subgraph(m).copy() for m in nx.connected_components(connections_graph))

    cube = np.zeros((3, 3, 3), dtype=np.int32, order="c")
    for k, sg in enumerate(subgraphs):
        for (x, y, z) in sg.nodes():
            cube[x, y, z] = k + 1

    cube_as_list = list(cube.ravel())

    return cube_as_list


def convert_cube_signature_to_bandage_array(cube_text):
    """
    Automatically detect the cube format signature (either hex or in list format)
    and return the cube in array[27] format
    If the cube is in the list format the separators between numbers can 
    be ".", ",", ";" and/or  space(s).
    example input string:
    "33EC01800846"
    or
    "3.4.5,6.1.2,7.1.2, 3.4.5,6.1.8,7.1.8, 9.9.0,10.10.11,12.12.11"
    """
    cube_text = cube_text.strip()
    cube_text_items = [m for m in re.split("\.|,|;| ", cube_text) if m]
    cube = None
    if len(cube_text_items) == 1:
        # hex signature
        cube_hex_signature = cube_text_items[0]
        cube = convert_hex_signature_to_bandage_array(cube_hex_signature)

    if len(cube_text_items) == 27:
        # cube in list format
        cube = [int(m) for m in cube_text_items]

    if not cube:
        raise ValueError("Could not recognize cube format:" + cube_text)
    return cube


def invert_tuple_list_to_dict(tuple_list):
    """
    Make a dictionary from the list of tuples where the second item every tuple is considered as key and the first item of every tuple is added to the value array
    Args:
      tuple_list: A list of tuple (pairs)
    Returns:
      A dictionary of list
    """
    result = {}
    for k, v in tuple_list:
        result.setdefault(v, list()).append(k)
    return result


def separate_nodes_by_categories(graph, max_number_nodes_per_category):
    """
       Separates the nodes by the category using ther degree
       The categories are ordered by importance and they are:
         - "cube" (most important with the highest degree, drawn as complete cube)
         - "circle_with_label" (if there are multiple node degrees which fits this category then it is drawn with multiple sizes depending the node level)
         - "label_only"
         - "circle
         - "none" (nodes with the lowest degrees)
       
       Args:
         graph: networkx graph
         max_number_nodes_per_category: A dictionary with keys as categories names and the value of the maximum nodes allowed for that category

       If there are too many nodes with the same degree then the all these nodes are put in the next category. This avoid having nodes with the same degree on multiple categories.
    """
    node_categories_thresholds_names = [
        "cube",
        "circle_with_label",
        "label_only",
        "circle",
    ]

    node_categories_current_threshold = 0
    node_categories_thresholds = []

    nodes_degrees = list(nx.degree(graph))
    for node_category_priority_name in node_categories_thresholds_names:
        number_nodes_per_category = max_number_nodes_per_category.get(
            node_category_priority_name, 0
        )
        node_categories_current_threshold += number_nodes_per_category
        node_categories_thresholds.append(node_categories_current_threshold)

    nodes_degrees_inv = invert_tuple_list_to_dict(nodes_degrees)
    degrees_with_number_of_nodes = [(k, len(v)) for (k, v) in nodes_degrees_inv.items()]

    total_nodes_sorted_by_degrees = {}
    total_nodes_by_degree = 0
    nodes_degree_categories = {}
    for (degree, degree_nodes) in sorted(
        degrees_with_number_of_nodes, key=lambda m: m[0], reverse=True
    ):
        total_nodes_by_degree += degree_nodes
        category_k = len(
            [m for m in node_categories_thresholds if m < total_nodes_by_degree]
        )
        try:
            category_name = node_categories_thresholds_names[category_k]
        except IndexError:
            category_name = "none"
        nodes_degree_categories[degree] = category_name

    return (nodes_degree_categories, nodes_degrees_inv)


def process_nodes(
    graph,
    i2c,
    svg_filename_prefix,
    nodes_degree_categories,
    nodes_degrees_inv,
    cube_draw_projection,
):
    """
       Process the graph nodes and set their attributes for graphviz

       Args:
           graph: the networkx graph (used for input and output)
           i2c: the dictionary with the cube representation where the key is the node number and the value is the cube array
           svg_filename_prefix: the file prefix of the generated SVG files
           nodes_degree_categories: a dictionary where the keys are the node degree and the value is the category name for that degree
           nodes_degrees_inv: a dictionary where the keys are the node degree and the values a list of all nodes with that degree
           cube_draw_projection: the cube projection for drawing

       Returns:
          None but it modifies the graph

    """

    circle_degree_categories_values = sorted(
        [k for (k, v) in nodes_degree_categories.items() if v == "circle"], reverse=True
    )
    circle_degree_k = {v: k for k, v in enumerate(circle_degree_categories_values)}
    full_cube_draw_node_list = []
    for degree, degree_category in sorted(
        nodes_degree_categories.items(), key=lambda m: m[0], reverse=True
    ):
        node_list = nodes_degrees_inv[degree]

        new_attributes = {}
        if degree_category == "cube":
            image_size = 100.0
            full_cube_draw_node_list += [(m, image_size) for m in node_list]
            new_attributes["shape"] = "none"
            new_attributes["label"] = ""
            new_attributes["height"] = "1"
            new_attributes["width"] = "1"
        elif degree_category == "circle_with_label":
            new_attributes["fontsize"] = "20"
            new_attributes["width"] = "0.5"
            new_attributes["height"] = "0.5"
            new_attributes["fixedsize"] = "true"
        elif degree_category == "label_only":
            new_attributes["fontsize"] = "15"
            new_attributes["width"] = "0.3"
            new_attributes["height"] = "0.3"
            new_attributes["penwidth"] = "0"
            new_attributes["bgcolor"] = "transparent"
        elif degree_category == "circle":
            new_attributes["shape"] = "circle"
            new_attributes["color"] = "black"
            new_attributes["style"] = "filled"
            new_attributes["label"] = ""
            new_attributes["fixedsize"] = "true"

            BASE_CIRCLE_WIDTH = 0.2
            cdk = min(circle_degree_k[degree], 4)
            width = BASE_CIRCLE_WIDTH * pow(0.5, cdk)
            new_attributes["width"] = "{}".format(width)
        else:
            new_attributes["width"] = "0.03"
            new_attributes["shape"] = "point"
            new_attributes["color"] = "#00000080"

        nodes_with_new_attributes = {k: new_attributes for k in node_list}
        nx.set_node_attributes(graph, nodes_with_new_attributes)

    # draw cubes svg
    for k_node, image_size in full_cube_draw_node_list:
        svg_filename = svg_filename_prefix + str(k_node) + ".svg"
        cube = i2c[k_node]
        draw_svg_cube(
            svg_filename=svg_filename,
            cube_order=3,
            cube=cube,
            image_size=image_size,
            color_mode="center",
            cube_draw_projection=cube_draw_projection,
        )
        nx.set_node_attributes(graph, {k_node: {"image": svg_filename}})


def process_edges(graph, edge_labels, show_labels=True, show_arrows=True):
    """
    Process the edges of the graph

    Args:
       graph: the networkx graph (used for input and output)
       edge_labels: a dictionary with keys as pairs of nodes and the value the edge label
                    A signle edge label can contain multiple letters which represents multiple face movements between the same nodes. They are displyed with the label "*"
       show_labels: if the labels of the edges is displayed
       show_arrows: if the arrows of the edges is displayed

    """
    labels_inv = invert_tuple_list_to_dict(edge_labels.items())
    faces_moves_color_map = FACE_COLORS

    for label, edge_list in labels_inv.items():
        col = faces_moves_color_map.get(label, UNKNOWN_FACE_EDGE_COLOR)
        new_attributes = {
            "color": col,
            "arrowsize": 1.0,
            "fontcolor": col,
            "fontsize": 16,
        }
        if show_labels:
            edge_label = label
            if label not in faces_moves_color_map:
                edge_label = "*"
            new_attributes["label"] = edge_label
        if not show_arrows:
            new_attributes["arrowhead"] = "none"
        nodes_with_new_attributes = {k: new_attributes for k in edge_list}
        nx.set_edge_attributes(graph, nodes_with_new_attributes)


def draw_legend(
    dot_graph,
    svg_filename_prefix,
    nodes_degree_categories,
    nodes_degrees_inv,
    i2c,
    cube_draw_projection,
):
    """
    Draw the graph legend with the cube starting position and the index for the cubes which are "circle_with_label"
    Args:
      dot_graph: pydot graph (used for output)
      svg_filename_prefix:  the file prefix of the generated SVG files
      nodes_degree_categories: a dictionary where the keys are the node degree and the value is the category name for that degree
      nodes_degrees_inv: a dictionary where the keys are the node degree and the values a list of all nodes with that degree
      i2c: the dictionary with the cube representation where the key is the node number and the value is the cube array
      cube_draw_projection: the cube projection for drawing

    Returns: None
    """

    LEGEND_CONFIG = {"cube_index_rows": 8, "max_allowed_index_size": 200}

    graph_legend = pydot.Cluster(label="", fontsize="16", rankdir="BT")

    # draw index
    all_nodes_for_index = []
    for node_degree, node_category in nodes_degree_categories.items():
        if node_category == "circle_with_label":
            all_nodes_for_index += nodes_degrees_inv[node_degree]
    all_nodes_for_index = sorted(all_nodes_for_index)
    image_size = 80

    index_node = None
    if 0 < len(all_nodes_for_index) <= LEGEND_CONFIG["max_allowed_index_size"]:
        column_size = len(all_nodes_for_index) // LEGEND_CONFIG["cube_index_rows"]
        if column_size == 0:
            column_size = 1
        index_nodes_rows = [
            all_nodes_for_index[k : k + column_size]
            for k in range(0, len(all_nodes_for_index), column_size)
        ]

        html_index_table = ["<table color='grey'>"]
        for index_node_row in index_nodes_rows:
            html_index_table += ["<tr>"]
            for index_node in index_node_row:
                if index_node is None:
                    continue
                index_node_filename = svg_filename_prefix + str(index_node) + ".svg"
                cube = i2c[index_node]
                draw_svg_cube(
                    svg_filename=index_node_filename,
                    cube_order=3,
                    cube=cube,
                    image_size=image_size,
                    color_mode="center",
                    label=str(index_node),
                    cube_draw_projection=cube_draw_projection,
                )
                html_index_table += [
                    "<td><img src='{}'/></td>".format(index_node_filename)
                ]
            html_index_table += ["</tr>"]

        html_index_table += ["</table>"]

        html_index_table = " ".join(html_index_table)
        index_node = pydot.Node(
            "index", shape="none", label="< {} >".format(html_index_table), rank="max"
        )
        graph_legend.add_node(index_node)

    # add start cube image
    node_filename = svg_filename_prefix + "start.svg"
    draw_svg_cube(
        svg_filename=node_filename,
        cube_order=3,
        cube=i2c[0],
        image_size=image_size * 3,
        color_mode="full",
        cube_draw_projection=cube_draw_projection,
    )

    start_node = pydot.Node(
        "start", shape="none", label="", rank="min", image=node_filename
    )

    dot_graph.add_node(start_node)
    dot_graph.add_subgraph(graph_legend)

    if index_node:
        dot_graph.add_edge(pydot.Edge(index_node, start_node, style="invis"))


def draw_cube_graph(
    explored_cube,
    cube_signature,
    cube_label,
    output_filename,
    skip_legend_draw,
    output_temporary_folder=None,
    cube_draw_projection=None,
):
    """
    Draw the cube graph and the legend

    Args:
        explored_cube: the bandage cube array with 27 cubies
        cube_signature: the hex cube signature
        cube_label: the name of the cube
        output_filename: the file output (tested with extensions ".svg", ".png", ".pdf")
        skip_legend_draw: the the legend is not being drawn
        output_temporary_folder: if set to a string, it ouputs the temporary svg files and saves the dot files to that directory. If False no temporary files are kept
        cube_draw_projection: the cube projection for drawing

    WARNING:
        If the legend is being drawn (skip_legend_draw is False) and the graph is very complex (like more than few tousands of nodes) the graph in the final image is will be empty
        That is due to the an issue of graphviz with loading of large SVG images
        Please read more detailed comment about this issue in this function.

    Return:
        None

    """
    nodes, edges, labels, i2c = explored_cube

    if not skip_legend_draw and len(nodes) > 10000:
        # more information about this issue is documented into the code below
        print(
            "Warning(bug): it is very likely to get empty graph image if skip_legend_draw is false on larger graphs. If you get this issue, as a workaround, enable skip_legend_draw"
        )

    g = nx.DiGraph(edges)

    tempfile_dir = None
    if output_temporary_folder:
        os.makedirs(output_temporary_folder, exist_ok=True)
    tempfile_dir = tempfile.TemporaryDirectory(prefix="bandage_cube_tmp")

    tmp_file_prefix = os.path.abspath(os.path.join(tempfile_dir.name, cube_signature))

    nodes_degree_categories, nodes_degrees_inv = separate_nodes_by_categories(
        g, MAX_NUMBER_OF_NODES_PER_CATEGORY
    )
    process_nodes(
        g,
        i2c,
        tmp_file_prefix + "_node_",
        nodes_degree_categories,
        nodes_degrees_inv,
        cube_draw_projection,
    )
    process_edges(
        g, labels, len(edges) <= SHOW_EDGE_LABELS_MAX, len(edges) < SHOW_EDGE_ARROWS_MAX
    )

    dot_graph = nx.nx_pydot.to_pydot(g)
    dot_graph_attributes = dot_graph.get_attributes()
    dot_graph_attributes["overlap"] = "false"
    dot_graph_attributes["label"] = cube_label
    dot_graph_attributes["labelloc"] = "top"
    dot_graph_attributes["dpi"] = "100"
    dot_graph_attributes["layout"] = "sfdp"

    # show the start node
    dot_graph_start_pointer = pydot.Node(
        "start_pointer", label="", shape="point", height=0, width=0
    )
    dot_graph.add_node(dot_graph_start_pointer)
    dot_graph.add_edge(pydot.Edge(dot_graph_start_pointer, "0", penwidth=2, label=""))

    file_ext = os.path.splitext(output_filename)[1].strip(".")
    dot_main = None
    if skip_legend_draw:
        dot_graph.write(output_filename, format="{}:cairo".format(file_ext))
    else:
        # Warning: very large graphs can't be embedded due to rsvg error
        # The error is "rsvg_handle_write returned an error: Error domain 1 code 1 on line 39037 column 1 of data: internal error: Huge input lookup"
        # and graphviz fails silently (an empty image file is used instead)
        # I picked the solution of multi-step graph rendering and loading of image
        # (instead of letting the graphviz to do it at once because
        # I couldn't find a way to specify a way to layout the graph
        # with sfdp and put the index nicely
        # A workaround is to skip drawing of the legend (skip_legend_draw=True)

        dot_graph_svg = tmp_file_prefix + "_graph.svg"
        dot_graph.write(dot_graph_svg, format="svg:cairo")

        # output the legend
        dot_index = pydot.Dot()
        draw_legend(
            dot_index,
            tmp_file_prefix + "_index_",
            nodes_degree_categories,
            nodes_degrees_inv,
            i2c,
            cube_draw_projection,
        )
        dot_index_attributes = dot_index.get_attributes()
        dot_index_attributes["layout"] = "dot"
        dot_index_attributes["overlap"] = "false"
        dot_index_attributes["dpi"] = "50"
        dot_index_svg = tmp_file_prefix + "_index.svg"
        dot_index.write(dot_index_svg, format="svg:cairo")

        # make the main graph which merges all the svg
        dot_main = pydot.Dot()
        dot_main_attributes = dot_main.get_attributes()
        dot_main_attributes["layout"] = "dot"
        dot_main_attributes["rankdir"] = "LR"
        dot_main_attributes["overlap"] = "false"
        dot_main_attributes["dpi"] = "200"

        graph_svg_node = pydot.Node(
            "graph_svg",
            shape="none",
            label="".format(cube_label),
            labelloc="top",
            rank="max",
            image=dot_graph_svg,
        )
        index_svg_node = pydot.Node(
            "index_svg", shape="none", label="", rank="min", image=dot_index_svg
        )
        dot_main.add_node(graph_svg_node)
        dot_main.add_node(index_svg_node)
        dot_main.add_edge(pydot.Edge(graph_svg_node, index_svg_node, style="invis"))

        dot_main.write(output_filename, format="{}:cairo".format(file_ext))

    if output_temporary_folder:
        # copy the temporary directory to the output_temporary_folder
        # todo: cleanup the next lines and output the svg links as relative path not as an absolute path of the old temporary directories
        output_tmp_file_prefix = os.path.join(
            output_temporary_folder, os.path.basename(tmp_file_prefix)
        )
        for tmp_filename in glob.glob(os.path.join(tempfile_dir.name, "*")):
            shutil.copy2(
                tmp_filename,
                os.path.join(output_temporary_folder, os.path.basename(tmp_filename)),
            )

        dot_graph.write(output_tmp_file_prefix + "_graph.dot", format="raw")
        if dot_main:
            dot_main.write(output_tmp_file_prefix + "_main.dot", format="raw")
        return tmp_file_prefix

    return None


def process_csv_file(
    csv_file_name,
    output_directory,
    file_extension="png",
    filter_by_number_of_nodes=None,
    skip_cubes_without_names=False,
    skip_legend_draw=False,
    cube_draw_projection=None,
):
    """
      process the csv file 
      It requires an csv file with columns named Hexa
      Optional columns: Name, N,E
      The meaning of the rows is:
          Hexa - cube hex signature
          Name - cube name
          N - number of nodes of the graph for cube
          E - number of edges of the graph for cube
      
      Args:
        csv_file_name: csv file path
        output_directory: the directory of output files
        file_extension: the extension and file format (examples: "png", "svg", "pdf")
        filter_by_number_of_nodes is a tuple of (min_nodes, max_nodes) for skipping processing of the cubes from the csv file
        skip_cubes_without_names: skip cubes without names
        skip_legend_draw: skip drawing of the legend for all cubes
        cube_draw_projection: the cube projection for drawing
    """
    with open(csv_file_name, "r") as f:
        csv_list = list(csv.reader(f))
        header = {m: k for k, m in enumerate(csv_list[0])}

        os.makedirs(output_directory, exist_ok=True)

        for row_k, csv_row in enumerate(csv_list[1:]):
            cube_signature = csv_row[header["Hexa"]].strip()
            cube_name = csv_row[header["Name"]].strip()
            if not cube_signature:
                continue

            if skip_cubes_without_names and not cube_name:
                continue

            try:
                cube_N = int(csv_row[header["N"]])
                cube_E = int(csv_row[header["E"]])
            except (ValueError, KeyError):
                cube_N = 0
                cube_E = 0

            try:
                cube = convert_cube_signature_to_bandage_array(cube_signature)
            except ValueError:
                print(" cube signature error:", cube_signature)
                continue

            explored_cube = None

            # try to filter based on the csv file information, otherwise explore the cube
            if cube_N == 0 or cube_E == 0:
                explored_cube = explore_cube(cube)
                cube_N, cube_E = len(explored_cube[0]), len(explored_cube[1])

            if filter_by_number_of_nodes:
                min_N, max_N = filter_by_number_of_nodes
                if cube_N < min_N or cube_N > max_N:
                    print(
                        " skipping cube {} with {} nodes".format(cube_signature, cube_N)
                    )
                    continue

            if not explored_cube:
                explored_cube = explore_cube(cube)

            nodes, edges, edge_labels, _ = explored_cube

            cube_N, cube_E = len(nodes), len(edges)

            print(
                "Cube:{}  nodes={} edges={} (csv row {}/{})".format(
                    cube_signature, cube_N, cube_E, row_k, len(csv_list) - 1
                )
            )

            output_filename = os.path.join(
                output_directory, "{}.{}".format(cube_signature, file_extension)
            )
            cube_label = "{} - {} (N={} E={}) ".format(
                cube_signature, cube_name, cube_N, cube_E
            )

            print("  Output file: ", output_filename)
            draw_cube_graph(
                explored_cube,
                cube_signature,
                cube_label,
                output_filename,
                skip_legend_draw=skip_legend_draw,
                cube_draw_projection=cube_draw_projection,
            )


def process_single_cube(
    cube_signature, output_filename, skip_legend_draw, cube_draw_projection
):
    """
    Process a single cube signature
    Args:
       cube_signature: the cube signature
       output_filename: the filename of the generated image
       skip_legend_draw: the the legend is not drawn
       cube_draw_projection: the cube projection for drawing
    """
    try:
        cube = convert_cube_signature_to_bandage_array(cube_signature)
    except ValueError:
        print("hex signature error:", cube_signature)
        return
    explored_cube = explore_cube(cube)
    nodes, edges, edge_labels, _ = explored_cube
    cube_N, cube_E = len(nodes), len(edges)
    cube_label = "{} - (N={} E={}) ".format(cube_signature, cube_N, cube_E)
    print(" cube: ", cube_label)

    draw_cube_graph(
        explored_cube,
        cube_signature,
        cube_label,
        output_filename,
        skip_legend_draw=skip_legend_draw,
        cube_draw_projection=cube_draw_projection,
    )


def process_cube_list(
    cube_signature_list,
    output_directory,
    file_format,
    skip_legend_draw,
    cube_draw_projection,
):
    """
    Process a list of cube signature
    Args:
       cube_signature_list: the cube signature list
       output_directory: the output_directory
       file_format: the file extension ("png" or "pdf")
       skip_legend_draw: the the legend is not drawn
       cube_draw_projection: the cube projection for drawing
    """
    os.makedirs(output_directory, exist_ok=True)
    for cube_signature in cube_signature_list:
        output_filename = os.path.join(
            output_directory, "{}.{}".format(cube_signature, file_format)
        )
        print("Processing cube {}, file: {}".format(cube_signature, output_filename))
        process_single_cube(
            cube_signature, output_filename, skip_legend_draw, cube_draw_projection
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bandage cube grapher, written by Nasca Octavian Paul http://www.paulnasca.com"
    )
    parser.add_argument("--output_directory", default=".", help="output directory")
    parser.add_argument(
        "--file_format",
        default="png",
        help='output file format like "png", "pdf" or "svg".',
    )
    parser.add_argument(
        "--skip_legend_draw",
        default=False,
        action="store_true",
        help="skip drawing of the legend",
    )

    parser.add_argument(
        "--cube_draw_projection",
        default="cube_map",
        choices=["isometric", "cube_map"],
        help="The cube drawing projection.",
    )
    parser.add_argument(
        "--process_csv_file", help="process a csv file containing hex cube signatures"
    )
    parser.add_argument(
        "--skip_cubes_without_names",
        default=False,
        action="store_true",
        help="skip cubes without names from the csv file",
    )
    parser.add_argument(
        "--filter_by_number_of_nodes",
        default=None,
        help="process only cubes with number of nodes in range min-max. Example: 100-500  or  0-10000",
    )

    parser.add_argument("cube_signatures", nargs="*", help="cube hex or array[27] signatures")

    args = parser.parse_args()

    if args.cube_signatures:
        process_cube_list(
            args.cube_signatures,
            args.output_directory,
            args.file_format,
            args.skip_legend_draw,
            args.cube_draw_projection,
        )
    if args.process_csv_file:
        print("Processing csv file", args.process_csv_file)
        filter_by_number_of_nodes = None
        if args.filter_by_number_of_nodes:
            filter_by_number_of_nodes = [
                int(m) for m in args.filter_by_number_of_nodes.split("-")
            ]

        process_csv_file(
            args.process_csv_file,
            args.output_directory,
            args.file_format,
            filter_by_number_of_nodes,
            args.skip_cubes_without_names,
            args.skip_legend_draw,
            args.cube_draw_projection,
        )
    if len(sys.argv) < 2:
        parser.print_usage()
