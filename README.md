
<h2> Mathematical and Computational Concepts</h2>

<h3>Introduction</h3>
<p>This problem set explores various mathematical concepts related to coordinate transformations, geometric projections, and mesh generation. These tasks involve computational methods such as parallel transport, stereographic projection, and Delaunay triangulation, all of which are widely used in physics and data analysis.</p>

<hr>

<h2>Task 1: Coordinate Transformation and Parallel Transport</h2>

<h3>Mathematical Background</h3>
<ul>
  <li><b>Coordinate Systems:</b> The conversion between spherical, Cartesian, and cylindrical coordinates is crucial for physics simulations. Each coordinate system represents spatial positions differently:</li>
  <ul>
    <li>Spherical: \( (r, \theta, \phi) \) </li>
    <li>Cartesian: \( (x, y, z) \) </li>
    <li>Cylindrical: \( (\rho, \psi, z) \) </li>
  </ul>
  <li><b>Parallel Transport:</b> This involves moving a vector along a curved surface (e.g., a sphere) while preserving its inner product with the local tangent space. The process reveals the <i>holonomy</i>, which is a measure of the vector’s deviation after being transported around a closed loop.</li>
</ul>

<h3>Algorithms for Visualization</h3>
<ul>
  <li><b>Mesh Generation:</b> A sphere is typically represented using a discretized mesh, which requires defining a set of grid points and triangulating them.</li>
  <li><b>Vector Field Plotting:</b> The local orthonormal basis vectors need to be plotted at different locations on the sphere.</li>
  <li><b>Parallel Transport Animation:</b> The trajectory of the vector as it moves along the surface is visualized using an animated quiver plot.</li>
</ul>

<h3>Data Structures Used</h3>
<ul>
  <li><b>Numpy Arrays:</b> Efficient storage of coordinate transformations.</li>
  <li><b>Matplotlib (3D Plots):</b> Visualization of the coordinate transformations and parallel transport.</li>
</ul>

<h3>Applications in Physics and Data Analysis</h3>
<ul>
  <li>Parallel transport is a fundamental concept in **general relativity**, describing how vectors change in curved space.</li>
  <li>Coordinate transformations are used in **robot kinematics**, **fluid dynamics**, and **satellite orbit simulations**.</li>
</ul>

<hr>

<h2>Task 2: Geometric Transformations (Stereographic Projection)</h2>

<h3>Mathematical Background</h3>
<ul>
  <li><b>Stereographic Projection:</b> A function that maps a point \( (x, y, z) \) on a sphere to a 2D plane:</li>
  <p> \[
  P' = \frac{x \hat{x} + y \hat{y}}{1 - z}
  \] </p>
  <li>This projection is **conformal**, meaning it preserves angles but not distances.</li>
  <li><b>Great Circles and Geodesics:</b> The shortest paths on a sphere are mapped to circles or straight lines in the projection.</li>
</ul>

<h3>Algorithms for Visualization</h3>
<ul>
  <li><b>Sphere Mesh Generation:</b> A uniform mesh of points on the unit sphere is created.</li>
  <li><b>Projection Calculation:</b> Each point is mapped to the 2D plane using the stereographic formula.</li>
  <li><b>Contour Mapping:</b> Verifying conformality by ensuring that angles between intersecting curves remain the same after projection.</li>
</ul>

<h3>Data Structures Used</h3>
<ul>
  <li><b>Mesh Grids:</b> A grid-based representation of the sphere.</li>
  <li><b>Matplotlib Contours:</b> Used to visualize transformed curves.</li>
</ul>

<h3>Applications in Physics and Data Analysis</h3>
<ul>
  <li>Used in **complex analysis**, where conformal mappings help solve differential equations.</li>
  <li>Applied in **computer vision** and **VR rendering**, where 3D objects are mapped to 2D displays.</li>
  <li>Appears in **quantum mechanics**, where wavefunctions on curved spaces can be analyzed using projections.</li>
</ul>

<hr>

<h2>Task 3: Lifting Map and Delaunay Triangulation</h2>

<h3>Mathematical Background</h3>
<ul>
  <li><b>Delaunay Triangulation:</b> A method to divide a set of 2D points into triangles such that the circumcircle of each triangle does not contain any other points.</li>
  <li><b>Lifting Map:</b> A transformation that maps a 2D surface into 3D using a function like:</li>
  <p> \[
  z = f(x, y) = x^2 + y^2
  \] </p>
  <li><b>Surface Curvature:</b> The shape operator and fundamental forms describe the curvature of the lifted surface.</li>
</ul>

<h3>Algorithms for Visualization</h3>
<ul>
  <li><b>Convex Hull Computation:</b> The convex envelope of a point cloud is found.</li>
  <li><b>Heatmap Visualization:</b> The difference in triangle areas before and after lifting is visualized using a 2D heatmap.</li>
  <li><b>Curvature Estimation:</b> Principal, Gaussian, and mean curvatures are computed for each vertex.</li>
</ul>

<h3>Data Structures Used</h3>
<ul>
  <li><b>Scipy.spatial Delaunay:</b> Computes triangulations.</li>
  <li><b>Matplotlib 3D Surface Plots:</b> Used to visualize the lifted mesh.</li>
</ul>

<h3>Applications in Physics and Data Analysis</h3>
<ul>
  <li>Used in **general relativity** to model curved spacetime.</li>
  <li>Plays a role in **material science**, where surfaces of crystals and biological membranes are analyzed.</li>
  <li>Important in **3D reconstruction** techniques in data analysis and AI.</li>
</ul>

<hr>

