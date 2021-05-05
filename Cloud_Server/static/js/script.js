$(document).ready(function(){
    //connect to the socket server. 
    var socketA = io.connect('http://' + document.domain + ':' + location.port + '/river_level');
    //var socketA = io.connect(null, {port: 5001, rememberTransport: false});
    console.log(document.domain)

    $('#river_A_level_id').attr('src', '/static/river_imgs/river_A_level.jpg?a=' + Math.random());

    $('#river_B_level_id').attr('src', '/static/river_imgs/river_B_level.jpg?a=' + Math.random());

    //receive details from server
    socketA.on('river_level_in_meters', function(msg) {
        if (msg.number[2] === 0){
            console.log("Current River A Level:" + msg.number[0]);
            console.log("Current River B Level:" + msg.number[1]);

            if (msg.number[0] < 2.40){            
                $('#display_river_A_level').html('Current River Level at SHOPPING: ' +  msg.number[0].toString() + ' meters.');
            }else{
                $('#flood_alert').html('Flood Alert! Take shelter now!');
                $('body').css("background-color","red");
                alert("Flood Alert! Take shelter now!");
            }
            $('#river_A_level_id').attr('src', '/static/river_imgs/river_A_level.jpg?a=' + Math.random());

            if (msg.number[1] < 1.50){            
                $('#display_river_B_level').html('Current River Level at SESC: ' +  msg.number[1].toString() + ' meters.');
            }else{
                $('#flood_alert').html('Flood Alert! Take shelter now!');
                $('body').css("background-color","red");
                alert("Flood Alert! Take shelter now!");
            }
            $('#river_B_level_id').attr('src', '/static/river_imgs/river_B_level.jpg?a=' + Math.random());

            if (msg.number[0] < 2.40 && msg.number[1] < 1.50){
                $('body').css("background-color","white");
            }
        }else{
            $('#display_river_B_level').html('Sorry. Currently, the system does not work at night.');
            $('body').css("background-color","gray");
        }
    });

});


 
