// A script for simulating particles with WebGL

// Define globals
let gl = null;
let glCanvas = null;
let shaderProgram;

window.addEventListener("load", main, false);

function main(){
	
	//Gauss's Law (Maxell Equation)
	makeLaTexCanvas(String.raw`\iint _{\partial \Omega } \mathbf {E} \cdot \mathrm {d} \mathbf {S} ={\frac {1}{\varepsilon _{0}}}\iiint _{\Omega }\rho \,\mathrm {d} V`, render_join);
	//Gauss's Law For Magnetism (Maxell Equation)
	makeLaTexCanvas(String.raw`\iint _{\partial \Omega } \mathbf {B} \cdot \mathrm {d} \mathbf {S} =0`, render_join);
	//Faraday's Law (Maxell Equation)
	makeLaTexCanvas(String.raw`{\displaystyle \oint _{\partial \Sigma } \mathbf{E} \cdot \mathrm {d} \mathbf {l} =-{\frac {\mathrm {d} }{\mathrm {d} t}}\iint _{\Sigma }\mathbf {B} \cdot \mathrm {d} \mathbf {S} }`, render_join);
	//Ampere's Law (Maxell Equation)
	makeLaTexCanvas(String.raw`{\displaystyle {\begin{aligned}\oint _{\partial \Sigma }&\mathbf {B} \cdot \mathrm {d} {\mathbf {l}}=\mu _{0}\left(\iint _{\Sigma }\mathbf {J} \cdot \mathrm {d} \mathbf {S} +\varepsilon _{0}{\frac {\mathrm {d} }{\mathrm {d} t}}\iint _{\Sigma }\mathbf {E} \cdot \mathrm {d} \mathbf {S} \right)\\\end{aligned}}}`, render_join);
	
	// Navier-Stokes Equation
	makeLaTexCanvas(String.raw`{\displaystyle \rho {\frac {D\mathbf {u} }{Dt}}=\rho \left({\frac {\partial \mathbf {u} }{\partial t}}+\mathbf {u} \cdot \nabla \mathbf {u} \right)=-\nabla {\bar {p}}+\mu \,\nabla ^{2}\mathbf {u} +{\tfrac {1}{3}}\mu \,\nabla (\nabla \cdot \mathbf {u} )+\rho \mathbf {g}}`, render_join);
	// Schrodinger Equation
	makeLaTexCanvas(String.raw`{\displaystyle i\hbar {\frac {d}{dt}}\vert \Psi (t)\rangle ={\hat {H}}\vert \Psi (t)\rangle }`, render_join);
	// Einstein Field Equations
	makeLaTexCanvas(String.raw`{\displaystyle G_{\mu \nu }+\Lambda g_{\mu \nu }={\frac {8\pi G}{c^{4}}}T_{\mu \nu }}`, render_join);
	// Euler Lagrange Equation
	makeLaTexCanvas(String.raw`{\displaystyle L_{x}(t,q(t),{\dot {q}}(t))-{\frac {\mathrm {d} }{\mathrm {d} t}}L_{v}(t,q(t),{\dot {q}}(t))=0}`, render_join);
	// Rendering Equation
	makeLaTexCanvas(String.raw`{\displaystyle L_{\text{o}}(\mathbf {x} ,\omega _{\text{o}},\lambda ,t)=L_{\text{e}}(\mathbf {x} ,\omega _{\text{o}},\lambda ,t)\ +\int _{\Omega }f_{\text{r}}(\mathbf {x} ,\omega _{\text{i}},\omega _{\text{o}},\lambda ,t)L_{\text{i}}(\mathbf {x} ,\omega _{\text{i}},\lambda ,t)(\omega _{\text{i}}\cdot \mathbf {n} )\operatorname {d} \omega _{\text{i}}}`, render_join);
}
function smoothStep(x){
	if (x<0) return 0;
	if (x>1) return 1;
	return 3*x*x-2*x*x*x;
}


n_images = 9;
image_list = [];
function render_join(image){
	image_list.push(image);
	if (image_list.length == n_images) render(image_list);
}


function makeTextCanvas(text) {
	// Process text
	var fontSize = 20;
	text = text.split("\n");
	var height = fontSize * (text.length+3);
	var width = fontSize * 0.65 * Math.max(...text.map( line => line.length));

	var textCanvas = document.createElement("canvas").getContext("2d");
	textCanvas.canvas.width  = width;
	textCanvas.canvas.height = height;
	textCanvas.font = "20px monospace";
	textCanvas.fillStyle = "green";

	for (var i_line=0; i_line < text.length; i_line++){
		textCanvas.fillText(text[i_line], fontSize, fontSize*(i_line+2));
	}
	return textCanvas.canvas;
}

async function makeLaTexCanvas(LaTexSource, callback) {
	// Process text
	var wrapper = MathJax.tex2svg(`${LaTexSource}`, {em: 10, ex: 5,display: true});
	var svg = wrapper.querySelector("svg").outerHTML;
	var image = new Image();
	image.src = 'data:image/svg+xml;base64,' + window.btoa(unescape(encodeURIComponent(svg)));
	//console.log(image.src);
	
	image.onload = function() {
		var canvas = document.createElement('canvas');
		
		canvas.width = image.width;
		canvas.height = image.height;
		var context = canvas.getContext('2d');
		context.filter = 'invert(1)'
		context.drawImage(image, 0, 0);

		//document.querySelector("#test").appendChild(canvas);
		callback(canvas);
	}
}


// The programs
let vertexShader = `
	const mat4 perspective_transform = mat4(
		1,0,0,0,
		0,-1,0,0,
		0,0,1,0,
		0,0,1,0
	);

	uniform float scroll_factor;

	uniform float azimuthal_angle;
	uniform float polar_angle;
	uniform vec3 translation;


	attribute vec2 initial_position;
	attribute vec2 a_texCoord;

	uniform vec3 clipspace_scale;

	varying vec2 v_texCoord;

	void main() {
		// First transform the position in viewspace
		vec3 viewspace_position = vec3(initial_position, 0);
		
		// Scroll factor
		viewspace_position.y += scroll_factor;
		// Azimuthal rotation
		float rad_azimuthal_angle = radians(azimuthal_angle);
		mat3 azimuthal_rotation = mat3(
			1, 0, 0,
			0, cos(rad_azimuthal_angle), -sin(rad_azimuthal_angle),
			0, sin(rad_azimuthal_angle), cos(rad_azimuthal_angle)
			);
		viewspace_position = viewspace_position * azimuthal_rotation;
		// Polar rotation
		float rad_polar_angle = radians(polar_angle);
		mat3 polar_rotation = mat3(
			cos(rad_polar_angle), 0, -sin(rad_polar_angle),
			0, 1, 0,
			sin(rad_polar_angle), 0, cos(rad_polar_angle)
			);
		viewspace_position = viewspace_position * polar_rotation;
		// Translation
		viewspace_position += translation;


		// Then set the clipspace position
		vec4 clipspace_position = vec4(2.0 * viewspace_position / clipspace_scale, 1);
		gl_Position = clipspace_position * perspective_transform;

		// pass the texCoord to the fragment shader
		// The GPU will interpolate this value between points.
		v_texCoord = a_texCoord;
	}
`;
let fragmentShader = `
	precision mediump float;

	// our texture
	uniform sampler2D u_image;
	// the texCoords passed in from the vertex shader.
	varying vec2 v_texCoord;

	uniform float alpha;


	void main() {
		vec4 pixel = texture2D(u_image, v_texCoord);
		gl_FragColor = vec4(pixel.rgb, alpha*pixel.a);
	}
`;


function render(image_list) {
	var image = image_list[0];

	// Initial setup
	glCanvas = document.getElementById("glcanvas");
	gl = glCanvas.getContext("webgl");
	// Set the view port
	gl.viewport(0,0, glCanvas.width, glCanvas.height);
	gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
	//gl.enable(gl.DITHER);
	gl.enable(gl.BLEND);
	//gl.enable(gl.DEPTH_TEST);
	//gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
	gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
	//gl.depthMask(false);
	
	// Compile the program
	program =  buildShaderProgram(vertexShader, fragmentShader);
	gl.useProgram(program);
	
	
	
	// Setup data
	var n_dim = 2;
	var n_tris = 2;
	
	// Uniform indicies
	var clipspace_scaleLocation = gl.getUniformLocation(program, "clipspace_scale");
	var scroll_factorLocation = gl.getUniformLocation(program, "scroll_factor");
	var alphaLocation = gl.getUniformLocation(program, "alpha");
	var azimuthal_angleLocation = gl.getUniformLocation(program, "azimuthal_angle");
	var polar_angleLocation = gl.getUniformLocation(program, "polar_angle");
	var translationLocation = gl.getUniformLocation(program, "translation");
	// Attributes indicies
	var positionLocation = gl.getAttribLocation(program, "initial_position");
	var texcoordLocation = gl.getAttribLocation(program, "a_texCoord");
	// Buffer allocation and indicies
	var positionBufferList = image_list.map(() => gl.createBuffer());
	var texcoordBuffer = gl.createBuffer();
	// Texture allocation and indicies
	var textureList = image_list.map(() => gl.createTexture());
	
	
	// Provide matching texture coordinates for the rectangle.
	var textureCoordinates = [
		0,  0,
		1,  0,
		0,  1,
		0,  1,
		1,  0,
		1,  1,
	];
	gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textureCoordinates), gl.STATIC_DRAW);


	for (var i=0; i < image_list.length; i++){
		// Create textures.
		gl.bindTexture(gl.TEXTURE_2D, textureList[i]);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image_list[i]);
		
		// Create a buffer to put the untransformed points in
		var rectangleCoordinates = [
			-image_list[i].width/2, -image_list[i].height/2,
			image_list[i].width/2, -image_list[i].height/2,
			-image_list[i].width/2, image_list[i].height/2,
			-image_list[i].width/2, image_list[i].height/2,
			image_list[i].width/2, -image_list[i].height/2,
			image_list[i].width/2, image_list[i].height/2,
		];
		gl.bindBuffer(gl.ARRAY_BUFFER, positionBufferList[i]);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(rectangleCoordinates), gl.STATIC_DRAW);
	}
	

	// Tell WebGL how to convert from clip space to pixels
	gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

	// Clear the canvas
	gl.clearColor(0, 0, 0, 1);
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

	// Turn on the texcoord attribute
	gl.enableVertexAttribArray(texcoordLocation);
	gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
	gl.vertexAttribPointer(texcoordLocation, n_dim, gl.FLOAT, false, 0, 0);

	// set the resolution
	gl.uniform3f(clipspace_scaleLocation, gl.canvas.width, gl.canvas.height, (gl.canvas.width+gl.canvas.height)/2);

	
	// Draw function
	function drawPlane(texture_index, alpha, scroll_factor, azimuthal_angle, polar_angle, x_translation, y_translation, z_translation){
		// Turn on the position attribute
		gl.enableVertexAttribArray(positionLocation);
		gl.bindBuffer(gl.ARRAY_BUFFER, positionBufferList[texture_index % textureList.length]);
		gl.vertexAttribPointer(positionLocation, n_dim, gl.FLOAT, false, 0, 0);

		// Prepare the texture
		gl.bindTexture(gl.TEXTURE_2D, textureList[texture_index % textureList.length]);
		
		// Set the transparency
		gl.uniform1f(alphaLocation, alpha);
	
		// set the transform
		gl.uniform1f(scroll_factorLocation, scroll_factor);
		gl.uniform1f(azimuthal_angleLocation, azimuthal_angle);
		gl.uniform1f(polar_angleLocation, polar_angle);
		gl.uniform3f(translationLocation, x_translation, y_translation, z_translation);

		// Draw the rectangle.
		gl.drawArrays(gl.TRIANGLES, 0, 3*n_tris);
	}

	
	// Define animation parameters
	var n_planes = 60;
	var speed = 100;
	// Inits
	var planeStack = [];
	for (var i_plane = 0; i_plane < n_planes; i_plane++) planeStack.push(
		{
			birthday: 0,
			azimuthal_angle: 0,
			polar_angle: 0,
			x_translation: 1000 * Math.random() - 500,
			y_translation: 1000 * Math.random() - 500,
			z_translation: 500 * Math.random()
		});


	// Setup animation
	var then = 0;
	var loopLength = 10;
	function drawFrame(time) {
		// Setup time delta
		time *= 0.001;
		var dt = time - then;
		then = time;
		var loopFraction = (time % loopLength)/parseFloat(loopLength);
		
		// Perform rendering
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

		for (var i = 0; i < planeStack.length; i++) {
			var i_plane = planeStack[i];
			
			// Move the plane closer
			i_plane.z_translation -= speed*dt;
			i_plane.z_translation = (500+i_plane.z_translation) % 500;
			var z_frac = (1-i_plane.z_translation/500);
			var alpha = smoothStep(2*z_frac)+smoothStep(2-2*z_frac)-1;
			
			drawPlane(
				i,
				alpha,
				0,//-0.5*(i_plane.z_translation - 1000),
				i_plane.azimuthal_angle,
				i_plane.polar_angle,
				i_plane.x_translation,
				i_plane.y_translation,
				i_plane.z_translation
				);
		}
		requestAnimationFrame(drawFrame);
	}
	requestAnimationFrame(drawFrame);
}




// Shader building tools from Mozilla
function buildShaderProgram(vertexSource, fragmentSource) {
	let program = gl.createProgram();
	
	// Compile the vertex shader
	let vShader = compileShader(vertexSource, gl.VERTEX_SHADER);
	if (vShader) gl.attachShader(program, vShader);
	// Compile the fragment shader
	let fShader = compileShader(fragmentSource, gl.FRAGMENT_SHADER);
	if (fShader) gl.attachShader(program, fShader);

	gl.linkProgram(program)

	if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
		console.log("Error linking shader program:");
		console.log(gl.getProgramInfoLog(program));
	}

	return program;
}

function compileShader(source, type) {
	let shader = gl.createShader(type);

	gl.shaderSource(shader, source);
	gl.compileShader(shader);

	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		console.log(`Error compiling ${type === gl.VERTEX_SHADER ? "vertex" : "fragment"} shader:`);
		console.log(gl.getShaderInfoLog(shader));
	}
	return shader;
}



