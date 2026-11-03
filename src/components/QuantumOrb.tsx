import { useEffect, useRef } from "react";
import * as THREE from "three";

interface QuantumOrbProps {
  intensity?: number;
  isSarcastic?: boolean | null;
}

export function QuantumOrb({ intensity = 0.5, isSarcastic = null }: QuantumOrbProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || window.innerWidth;
    const height = 110;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 4.0;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    // Color dynamic palette based on verdict
    const primaryColor = isSarcastic === true ? 0xf43f5e : isSarcastic === false ? 0x10b981 : 0x6366f1;
    const secondaryColor = isSarcastic === true ? 0xfb7185 : isSarcastic === false ? 0x34d399 : 0xa855f7;
    const accentColor = isSarcastic === true ? 0xfecdd3 : isSarcastic === false ? 0xa7f3d0 : 0x38bdf8;

    // Torus Rings
    const ringGeo1 = new THREE.TorusGeometry(1.4, 0.015, 16, 100);
    const ringMat1 = new THREE.MeshBasicMaterial({
      color: primaryColor,
      transparent: true,
      opacity: 0.75,
    });
    const ring1 = new THREE.Mesh(ringGeo1, ringMat1);
    group.add(ring1);

    const ringGeo2 = new THREE.TorusGeometry(1.15, 0.014, 16, 100);
    const ringMat2 = new THREE.MeshBasicMaterial({
      color: secondaryColor,
      transparent: true,
      opacity: 0.6,
    });
    const ring2 = new THREE.Mesh(ringGeo2, ringMat2);
    ring2.rotation.x = Math.PI / 3;
    group.add(ring2);

    const ringGeo3 = new THREE.TorusGeometry(0.85, 0.012, 16, 100);
    const ringMat3 = new THREE.MeshBasicMaterial({
      color: accentColor,
      transparent: true,
      opacity: 0.5,
    });
    const ring3 = new THREE.Mesh(ringGeo3, ringMat3);
    ring3.rotation.y = Math.PI / 4;
    group.add(ring3);

    // Ambient floating particles
    const particleCount = 140;
    const pGeo = new THREE.BufferGeometry();
    const pPos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i++) {
      pPos[i] = (Math.random() - 0.5) * 5.5;
    }
    pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
    const pMat = new THREE.PointsMaterial({
      size: 0.035,
      color: primaryColor,
      transparent: true,
      opacity: 0.65,
    });
    const particleMesh = new THREE.Points(pGeo, pMat);
    scene.add(particleMesh);

    let mouseX = 0;
    let mouseY = 0;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      mouseX = ((e.clientX - rect.left) / width - 0.5) * 2;
      mouseY = ((e.clientY - rect.top) / height - 0.5) * 2;
    };

    window.addEventListener("mousemove", handleMouseMove);

    let animId: number;
    const speedMult = 1 + (intensity || 0.5) * 1.5;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      ring1.rotation.x += 0.005 * speedMult;
      ring1.rotation.y += 0.007 * speedMult;
      ring2.rotation.y += 0.006 * speedMult;
      ring3.rotation.z += 0.008 * speedMult;

      group.rotation.x += (mouseY * 0.35 - group.rotation.x) * 0.05;
      group.rotation.y += (mouseX * 0.35 - group.rotation.y) * 0.05;

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      renderer.setSize(w, height);
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
    };

    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
      ringGeo1.dispose();
      ringMat1.dispose();
      ringGeo2.dispose();
      ringMat2.dispose();
      ringGeo3.dispose();
      ringMat3.dispose();
      pGeo.dispose();
      pMat.dispose();
      if (container) container.innerHTML = "";
    };
  }, [intensity, isSarcastic]);

  return (
    <div
      ref={containerRef}
      className="w-full h-[110px] overflow-hidden flex items-center justify-center pointer-events-none select-none"
    />
  );
}
