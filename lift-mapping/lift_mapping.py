import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

# Load the point cloud data from the .dat file, skipping the header row
points = np.loadtxt('mesh.dat', skiprows=1)

# Verify the loaded data
print("Loaded points shape:", points.shape)
print(points)

# Plot the Convex Hull
hull = ConvexHull(points)

# Create a new figure for the convex hull
plt.figure(figsize=(6, 6))
plt.plot(points[:, 0], points[:, 1], 'o', label='Points')  # Plot the points

# Plot the convex hull edges
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-', label='Convex Hull' if simplex[0] == 0 else "")

plt.title('2D Convex Hull')
plt.legend()
#plt.show() # Save the convex hull plot

# Generate and Plot the Delaunay Triangulation
tri = Delaunay(points)

# Create a new figure for the Delaunay triangulation
plt.figure(figsize=(6, 6))
plt.triplot(points[:, 0], points[:, 1], tri.simplices, 'k-', label='Delaunay Triangulation')  # Plot the triangulation
plt.plot(points[:, 0], points[:, 1], 'o', label='Points')  # Plot the points

plt.title('2D Delaunay Triangulation')
plt.legend()
#plt.show()

# Define the lifting map function: f(x, y) = x**2 + y**2
def lift_point(p):
    x, y = p
    return np.array([x, y, x**2 + y**2])

# Function to compute the area of a 2D triangle given its vertices
def triangle_area_2d(p1, p2, p3):
    # Using the cross product formula (the absolute value gives twice the area)
    return 0.5 * abs((p2[0] - p1[0])*(p3[1] - p1[1]) - (p3[0] - p1[0])*(p2[1] - p1[1]))

# Function to compute the area of a 3D triangle given its vertices
def triangle_area_3d(p1, p2, p3):
    # First lift the points to 3D
    p1_3d = lift_point(p1)
    p2_3d = lift_point(p2)
    p3_3d = lift_point(p3)
    # Compute two edge vectors
    v1 = p2_3d - p1_3d
    v2 = p3_3d - p1_3d
    # The area is 0.5 times the norm of the cross product of these edge vectors
    cross_prod = np.cross(v1, v2)
    return 0.5 * np.linalg.norm(cross_prod)

# For each triangle (each "simplex" in the Delaunay triangulation), compute the area ratio:
#   ratio = (area after lifting) / (area before lifting)
area_ratios = []
for simplex in tri.simplices:
    # Get the three vertices (each is a 2D point)
    p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
    area2d = triangle_area_2d(p1, p2, p3)
    area3d = triangle_area_3d(p1, p2, p3)
    # Protect against division by zero (should not occur if points are not collinear)
    ratio = area3d / area2d if area2d != 0 else 0
    area_ratios.append(ratio)
area_ratios = np.array(area_ratios)
# --- Plotting the Heatmap ---

plt.figure(figsize=(8, 6))
# Use tripcolor to plot a "heatmap" on the original x,y grid:
#  - The triangulation is defined by tri.simplices.
#  - facecolors are set to the area ratios computed.
tpc = plt.tripcolor(points[:, 0], points[:, 1], tri.simplices,
                    facecolors=area_ratios, edgecolors='k', cmap='plasma',clim=(0,50))

# Add a colorbar to display the area ratio scale
plt.colorbar(tpc, label='Area Ratio (3D / 2D)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Heatmap of Area Ratio after Lifting')
plt.tight_layout()
plt.savefig("/Users/yudonglei/Desktop/physics129_test/Pset2/lift-mapping/Heatmap of Area Ratio.png")
plt.show()

def induced_metrix(x, y):
    '''
    the induced matrix tensor at (x, y) for the surface: z = x**2 + y**2
    '''
    p_xx = 1 + 4 * x**2
    p_yy = 1 + 4 * y**2
    p_xy = 4 * x * y
    return np.array([[p_xx, p_xy],
                     [p_xy, p_yy]])

x, y = 10, 10
m = induced_metrix(x, y)
print("Induced metrix at (10, 10): \n", m)

X = points[:, 0]
Y = points[:, 1]
Z = X**2 + Y**2
points3d = np.column_stack((X, Y, Z))


# Compute the surface normals at each point

# For a surface z = f(x,y), the normal (not necessarily unit-length) is given by (-f_x, -f_y, 1).
# For f(x,y) = x^2+y^2, we have f_x = 2x and f_y = 2y.
# Thus, the (unit) normal is:
#     n = (-2x, -2y, 1) / sqrt(4x^2+4y^2+1)

# Compute the denominator (normalization factor) for each point:
denom = np.sqrt((2*X)**2 + (2*Y)**2 + 1)
normals = np.empty((len(X), 3))
normals[:, 0] = -2 * X / denom  # n_x
normals[:, 1] = -2 * Y / denom  # n_y
normals[:, 2] = 1 / denom       # n_z

face_centroids = []
face_normals = []
for triangle in tri.simplices:
    # Get indices for the three vertices of the triangle
    i, j, k = triangle
    
    # Retrieve the (x,y) coordinates for each vertex
    x0, y0, z0 = points3d[i]
    x1, y1, z1 = points3d[j]
    x2, y2, z2 = points3d[k]
    
    # Compute the centroid in the (x,y) plane
    x_c = (x0 + x1 + x2) / 3.0
    y_c = (y0 + y1 + y2) / 3.0
    # Lift the centroid to 3D
    z_c = (z0 + z1 + z2) / 3.0
    centroid = np.array([x_c, y_c, z_c])
    face_centroids.append(centroid)
    
    # Compute the analytic surface normal at (x_c, y_c)
    # For z = x^2 + y^2, we have f_x = 2x and f_y = 2y, so the (unnormalized) normal is (-2x, -2y, 1).
    denom = np.sqrt(4*x_c**2 + 4*y_c**2 + 1)
    n_face = np.array([2*x_c, 2*y_c, -1]) / denom
    face_normals.append(n_face)

face_centroids = np.array(face_centroids)
face_normals = np.array(face_normals)

# Compute vertex normals by averaging the face normals of adjacent triangles

num_vertices = len(points)
vertex_normals = np.zeros((num_vertices, 3))

# For each vertex, find all triangles (by index) that include this vertex,
# average their face normals, and then normalize the result.
for v in range(num_vertices):
    adjacent_normals = []
    for i, triangle in enumerate(tri.simplices):
        if v in triangle:
            adjacent_normals.append(face_normals[i])
    if adjacent_normals:
        adjacent_normals = np.array(adjacent_normals)
        avg_normal = np.mean(adjacent_normals, axis=0)
        norm_val = np.linalg.norm(avg_normal)
        if norm_val > 0:
            avg_normal /= norm_val
        vertex_normals[v] = avg_normal

# -----------------------------
# 4. Plot the lifted mesh and overlay the surface normals
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh surface using plot_trisurf with the Delaunay triangles:
ax.plot_trisurf(X, Y, Z, triangles=tri.simplices, cmap='plasma',
                alpha=0.5, edgecolor='k')

# For clarity, plot the normals for a subset of points
# (if your mesh has many points, plotting all normals might be too cluttered)
#skip = 1  # adjust this value as needed to reduce clutter
ax.quiver(face_centroids[:, 0], face_centroids[:, 1], face_centroids[:, 2],
          -1*face_normals[:, 0], -1*face_normals[:, 1], -1*face_normals[:, 2],
          length=0.5, color='blue', label='Face Normals')

# Label the axes and add a title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lifted Mesh with Surface Normals')

plt.tight_layout()
plt.savefig("/Users/yudonglei/Desktop/physics129_test/Pset2/lift-mapping/Lifted Mesh With Surface Norms.png")
plt.show()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh surface using plot_trisurf with the Delaunay triangles:
ax.plot_trisurf(X, Y, Z, triangles=tri.simplices, cmap='plasma',
                alpha=0.5, edgecolor='k')

# For clarity, plot the normals for a subset of points
# (if your mesh has many points, plotting all normals might be too cluttered)
skip = 1  # adjust this value as needed to reduce clutter

ax.quiver(points3d[:, 0], points3d[:, 1], points3d[:, 2],
          -1*vertex_normals[:, 0], -1*vertex_normals[:, 1], -1*vertex_normals[:, 2],
          length=0.5, color='blue', label='Vertex Normals')

# Label the axes and add a title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lifted Mesh with Vertex Normals')

plt.tight_layout()
plt.savefig("/Users/yudonglei/Desktop/physics129_test/Pset2/lift-mapping/Lifted Mesh with Vertex Norms.png")
plt.show()

# Compute the second fundamental form using the computed vertex normals.

# For z = x^2+y^2, the analytical second derivatives are:
#   X_xx = (0,0,2),   X_xy = (0,0,0),   X_yy = (0,0,2).
# For a vertex with vertex normal n_v = (n_x, n_y, n_z),
#   L = <X_xx, n_v> = 2 * n_z,
#   M = <X_xy, n_v> = 0,
#   N = <X_yy, n_v> = 2 * n_z.

L = 2 * vertex_normals[:, 2]  # L for each vertex
M = np.zeros_like(L)            # M is zero at every vertex
N = 2 * vertex_normals[:, 2]    # N for each vertex

# For demonstration, print the second fundamental form coefficients for the first 10 vertices:
print("Second Fundamental Form Coefficients (per vertex):")
for i in range(len(points)):
    print(f"Vertex {i}: L = {L[i]:.4f}, M = {M[i]:.4f}, N = {N[i]:.4f}")

k1s = []   # principal curvature 1
k2s = []   # principal curvature 2
Gauss = [] # Gaussian curvature
Mean = []  # Mean curvature

for i, (x, y) in enumerate(points):
    r2 = x**2 + y**2
    # First fundamental form I:
    E = 1 + 4*x**2
    F = 4*x*y
    G = 1 + 4*y**2
    I_mat = np.array([[E, F],
                      [F, G]])
    detI = E*G - F**2  # = 1 + 4*(x^2+y^2)
    # Inverse of I:
    I_inv = np.array([[G, -F],
                      [-F, E]]) / detI

    # Second fundamental form II:
    # Using analytic second derivatives and the vertex normal's z-component:
    n_z = vertex_normals[i, 2]  # should equal 1/sqrt(1+4r2)
    L = 2 * n_z
    M = 0
    N = 2 * n_z
    II_mat = np.array([[L, M],
                       [M, N]])

    # Shape operator S = I^{-1} II (a 2x2 matrix)
    S = I_inv @ II_mat

    # Diagonalize S to obtain its eigenvalues (principal curvatures)
    eigvals = np.linalg.eigvals(S)
    # Sort so that k1 >= k2
    eigvals = np.sort(eigvals)[::-1]
    k1 = eigvals[0]
    k2 = eigvals[1]
    k1s.append(k1)
    k2s.append(k2)
    Gauss.append(k1 * k2)
    Mean.append((k1 + k2) / 2)

k1s = np.array(k1s)
k2s = np.array(k2s)
Gauss = np.array(Gauss)
Mean = np.array(Mean)

#  Visualization: Scatter Plots of the Curvatures on the (x,y) Domain

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

sc0 = axs[0, 0].scatter(X, Y, c=k1s, cmap='plasma')
axs[0, 0].set_title('Principal Curvature k₁')
fig.colorbar(sc0, ax=axs[0, 0])
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')

sc1 = axs[0, 1].scatter(X, Y, c=k2s, cmap='plasma')
axs[0, 1].set_title('Principal Curvature k₂')
fig.colorbar(sc1, ax=axs[0, 1])
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('y')

sc2 = axs[1, 0].scatter(X, Y, c=Gauss, cmap='plasma')
axs[1, 0].set_title('Gaussian Curvature')
fig.colorbar(sc2, ax=axs[1, 0])
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('y')

sc3 = axs[1, 1].scatter(X, Y, c=Mean, cmap='plasma')
axs[1, 1].set_title('Mean Curvature')
fig.colorbar(sc3, ax=axs[1, 1])
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('y')

plt.suptitle('Curvature Visualizations on the Lifted Surface (z = x²+y²)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/Users/yudonglei/Desktop/physics129_test/Pset2/lift-mapping/Curvature Visualization.png")
plt.show()