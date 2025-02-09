document.addEventListener("DOMContentLoaded", function() {
    fetch('/colormaps')
        .then(response => response.json())
        .then(data => {
            let colormapSelect = document.getElementById("colormap");
            data.colormaps.forEach(color => {
                let option = document.createElement("option");
                option.value = color;
                option.text = color;
                colormapSelect.appendChild(option);
            });
        });
});

function toggleColorbar() {
    let minInput = document.getElementById("colorbar_min");
    let maxInput = document.getElementById("colorbar_max");
    let checkbox = document.getElementById("auto_colorbar");

    if (checkbox.checked) {
        minInput.value = "";
        maxInput.value = "";
        minInput.disabled = true;
        maxInput.disabled = true;
    } else {
        minInput.disabled = false;
        maxInput.disabled = false;
    }
}

function toggleContour() {
    let contourInput = document.getElementById("contour_levels");
    let checkbox = document.getElementById("auto_contour");

    if (checkbox.checked) {
        contourInput.value = "";
        contourInput.disabled = true;
    } else {
        contourInput.disabled = false;
    }
}

function toggleColormap() {
    let colormapSelect = document.getElementById("colormap");
    let checkbox = document.getElementById("default_colormap");

    if (checkbox.checked) {
        colormapSelect.disabled = true;
    } else {
        colormapSelect.disabled = false;
    }
}


function getColorbarValues() {
    let minValue = document.getElementById("colorbar_min").value;
    let maxValue = document.getElementById("colorbar_max").value;
    let isAuto = document.getElementById("auto_colorbar").checked;

    return {
        colorbar_min: isAuto || minValue === "" ? null : parseInt(minValue),
        colorbar_max: isAuto || maxValue === "" ? null : parseInt(maxValue)
    };
}

function getContourLevels() {
    let contourValue = document.getElementById("contour_levels").value;
    let isAuto = document.getElementById("auto_contour").checked;

    return isAuto || contourValue === "" ? null : parseInt(contourValue);
}


function getColormap() {
    let colormapSelect = document.getElementById("colormap").value;
    let isAuto = document.getElementById("default_colormap").checked;

    return isAuto ? null : colormapSelect;
}


function selectDirectory() {
        fetch('/select_directory')
            .then(response => response.json())
            .then(data => {
                document.getElementById("save_path").value = data.path;
            });
    }

function plotGraph() {
    document.getElementById("status").innerText = "Processing...";

    let data = {
        longitude_lower: parseInt(document.getElementById("longitude_lower").value),
        longitude_upper: parseInt(document.getElementById("longitude_upper").value),
        latitude_lower: parseInt(document.getElementById("latitude_lower").value),
        latitude_upper: parseInt(document.getElementById("latitude_upper").value),
        longitude_res: parseFloat(document.getElementById("longitude_res").value),
        latitude_res: parseFloat(document.getElementById("latitude_res").value),
        save_path: document.getElementById("save_path").value,
//        colormap: document.getElementById("colormap").value,
        colormap: getColormap(),
        contour_levels: getContourLevels(),
        colorbar_min: getColorbarValues().colorbar_min,
        colorbar_max: getColorbarValues().colorbar_max
    };

    fetch('/plot', {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.error) {
            document.getElementById("status").innerText = "Error: " + result.error;
        } else {
            document.getElementById("status").innerText =
                "Execution Time: " + result.execution_time + " seconds";

            if (result.redirect_url) {
                window.location.href = result.redirect_url;
            }
        }
    })
    .catch(error => {
        document.getElementById("status").innerText = "Network error!";
    });
}
