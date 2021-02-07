import { Component, Input, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/environment';

class albumMeta{
  images?:number;
}

@Component({
  selector: 'app-album',
  templateUrl: './album.component.html',
  styleUrls: ['./album.component.css']
})
export class AlbumComponent implements OnInit {

  @Input() name:string | undefined;
  images:[string, boolean][] = [];
  
  carousel:Element | undefined;
  constructor(
    private route: ActivatedRoute,
    private http: HttpClient
  ) {
  }

  ngOnInit(): void {
    var carousel = document.querySelector('#carouselid')
    const routeParams = this.route.snapshot.paramMap;
    this.name = routeParams.get("name") || undefined;
    const url = environment.api+"/picture/"+this.name
    this.http.get( url).subscribe((resp:any)=>{
      this.images = new Array(resp.images).fill("").map((_, index)=>{
        return [environment.api+"/picture/"+this.name+"/"+index, index<3 || index>resp.images-2];
      })
      carousel?.addEventListener('slide.bs.carousel', (ev:any)=>{
        this.images[(ev.to-1)%this.images.length][1] = true;
        this.images[(ev.to)%this.images.length][1] = true;
        this.images[(ev.to+1)%this.images.length][1] = true;
      })
    })
  }

}
