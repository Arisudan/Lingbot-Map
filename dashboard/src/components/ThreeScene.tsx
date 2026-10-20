import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface ThreeSceneProps {
  points: number[][];
  colors: number[][];
  cameraRot: number[][];
  trajectory: number[][];
  alignment: {
    offset_x: number;
    offset_z: number;
    offset_y: number;
    scale: number;
    rotation: number;
  };
  show2DMap: boolean;
  mapUrl: string;
  mapBounds: {
    xMin: number;
    xMax: number;
    zMin: number;
    zMax: number;
  };
  viewDirection: string;
  onResetViewFinished: () => void;
}

export const ThreeScene: React.FC<ThreeSceneProps> = ({
  points,
  colors,
  cameraRot,
  trajectory,
  alignment,
  show2DMap,
  mapUrl,
  mapBounds,
  viewDirection,
  onResetViewFinished,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  
  // Point cloud elements
  const pointsGeometryRef = useRef<THREE.BufferGeometry | null>(null);
  const pointsMeshRef = useRef<THREE.Points | null>(null);
  
  // Trajectory elements
  const trajLineRef = useRef<THREE.Line | null>(null);
  const droneMeshRef = useRef<THREE.Mesh | null>(null);
  
  // 2D Map floor plane
  const floorPlaneRef = useRef<THREE.Mesh | null>(null);

  // Initialize Scene, Camera, Renderer, Controls
  useEffect(() => {
    if (!mountRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101217);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 15, 20);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    // Helpers (Grid and Axes)
    const gridHelper = new THREE.GridHelper(50, 50, 0x3c424f, 0x222630);
    gridHelper.position.y = -2;
    scene.add(gridHelper);

    // Resize Handler
    const handleResize = () => {
      if (!mountRef.current || !cameraRef.current) return;
      const w = mountRef.current.clientWidth;
      const h = mountRef.current.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    // Animation Loop
    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      if (controlsRef.current) controlsRef.current.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Update Point Cloud Data
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || points.length === 0) return;

    // Clean up old points
    if (pointsMeshRef.current) {
      scene.remove(pointsMeshRef.current);
      pointsGeometryRef.current?.dispose();
    }

    const geometry = new THREE.BufferGeometry();
    const positions: number[] = [];
    const colorsData: number[] = [];

    points.forEach((pt, i) => {
      positions.push(pt[0], pt[1], pt[2]);
      if (colors && colors[i]) {
        colorsData.push(colors[i][0], colors[i][1], colors[i][2]);
      } else {
        colorsData.push(0.8, 0.8, 0.8);
      }
    });

    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colorsData, 3));

    const material = new THREE.PointsMaterial({
      size: 0.12,
      vertexColors: true,
      sizeAttenuation: true,
    });

    const mesh = new THREE.Points(geometry, material);
    scene.add(mesh);
    pointsGeometryRef.current = geometry;
    pointsMeshRef.current = mesh;
  }, [points, colors]);

  // Update Trajectory and Drone Pointer
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    // Clean up old path
    if (trajLineRef.current) scene.remove(trajLineRef.current);
    if (droneMeshRef.current) scene.remove(droneMeshRef.current);

    if (trajectory.length > 0) {
      // 1. Draw green path
      const pointsArray = trajectory.map(pt => new THREE.Vector3(pt[0], pt[1], pt[2]));
      const geometry = new THREE.BufferGeometry().setFromPoints(pointsArray);
      const material = new THREE.LineBasicMaterial({ color: 0x5ace68, linewidth: 2 });
      const line = new THREE.Line(geometry, material);
      scene.add(line);
      trajLineRef.current = line;

      // 2. Draw Drone mesh pointer (represented by yellow cone pointing forward)
      const currentPos = trajectory[trajectory.length - 1];
      const coneGeom = new THREE.ConeGeometry(0.4, 1.2, 8);
      coneGeom.rotateX(Math.PI / 2); // Point forward
      const coneMat = new THREE.MeshBasicMaterial({ color: 0xfbdd0b });
      const drone = new THREE.Mesh(coneGeom, coneMat);
      drone.position.set(currentPos[0], currentPos[1], currentPos[2]);
      
      // Orient camera according to cameraRot if available
      if (cameraRot && cameraRot.length === 3) {
        const matrix = new THREE.Matrix4();
        matrix.set(
          cameraRot[0][0], cameraRot[0][1], cameraRot[0][2], 0,
          cameraRot[1][0], cameraRot[1][1], cameraRot[1][2], 0,
          cameraRot[2][0], cameraRot[2][1], cameraRot[2][2], 0,
          0, 0, 0, 1
        );
        drone.rotation.setFromRotationMatrix(matrix);
      }
      scene.add(drone);
      droneMeshRef.current = drone;
    }
  }, [trajectory, cameraRot]);

  // Update 2D Floor Map texture and alignment transformation
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (floorPlaneRef.current) {
      scene.remove(floorPlaneRef.current);
      floorPlaneRef.current = null;
    }

    if (!show2DMap || !mapUrl) return;

    const x_center = (mapBounds.xMin + mapBounds.xMax) / 2.0 + alignment.offset_x;
    const z_center = (mapBounds.zMin + mapBounds.zMax) / 2.0 + alignment.offset_z;
    const y_level = alignment.offset_y;

    const width = (mapBounds.xMax - mapBounds.xMin) * alignment.scale;
    const height = (mapBounds.zMax - mapBounds.zMin) * alignment.scale;

    // Load dynamic map URL texture
    const texture = new THREE.TextureLoader().load(mapUrl);
    texture.minFilter = THREE.LinearFilter;
    
    const geom = new THREE.PlaneGeometry(width, height);
    const mat = new THREE.MeshBasicMaterial({
      map: texture,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
    });

    const mesh = new THREE.Mesh(geom, mat);
    
    // Position plane at aligned coordinates
    mesh.position.set(x_center, y_level, z_center);
    
    // Rotate plane to lie flat on X-Z plane, and apply Yaw alignment rotation
    mesh.rotation.x = -Math.PI / 2; // Flat on horizontal plane
    mesh.rotation.z = (alignment.rotation * Math.PI) / 180; // Yaw rotation around vertical axis

    scene.add(mesh);
    floorPlaneRef.current = mesh;
  }, [show2DMap, mapUrl, mapBounds, alignment]);

  // Camera Reset / View presets handler
  useEffect(() => {
    if (!cameraRef.current || !controlsRef.current || !viewDirection) return;

    const camera = cameraRef.current;
    const controls = controlsRef.current;

    // Compute bounding center of point cloud
    let targetLookAt = new THREE.Vector3(0, 0, 0);
    if (trajectory.length > 0) {
      const currentPos = trajectory[trajectory.length - 1];
      targetLookAt.set(currentPos[0], currentPos[1], currentPos[2]);
    }

    if (viewDirection === "center") {
      controls.target.copy(targetLookAt);
      camera.position.set(targetLookAt.x, targetLookAt.y + 15, targetLookAt.z + 20);
    } else if (viewDirection === "overview") {
      controls.target.copy(targetLookAt);
      camera.position.set(targetLookAt.x - 20, targetLookAt.y + 25, targetLookAt.z + 30);
    } else if (viewDirection === "front") {
      controls.target.copy(targetLookAt);
      camera.position.set(targetLookAt.x, targetLookAt.y, targetLookAt.z + 25);
    } else if (viewDirection === "back") {
      controls.target.copy(targetLookAt);
      camera.position.set(targetLookAt.x, targetLookAt.y, targetLookAt.z - 25);
    } else if (viewDirection === "top") {
      controls.target.copy(targetLookAt);
      camera.position.set(targetLookAt.x, targetLookAt.y + 35, targetLookAt.z);
    } else if (viewDirection === "left") {
      controls.target.copy(targetLookAt);
      camera.position.set(targetLookAt.x - 25, targetLookAt.y, targetLookAt.z);
    } else if (viewDirection === "right") {
      controls.target.copy(targetLookAt);
      camera.position.set(targetLookAt.x + 25, targetLookAt.y, targetLookAt.z);
    }

    controls.update();
    onResetViewFinished();
  }, [viewDirection, trajectory, onResetViewFinished]);

  return <div ref={mountRef} className="w-full h-full relative" />;
};
